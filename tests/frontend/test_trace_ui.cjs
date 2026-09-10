// 验证轨迹状态与流读取；不访问模型、不依赖浏览器或第三方测试包。
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { toolPresentation, upsertTool, reconcileTools, consumeAgentStream } = require('../../app/static/pilot_demo/app.js');

test('grep 实时展示名称、关键词、命中行数和截断状态', () => {
  const traces = [];
  upsertTool(traces, {tool_call_id: 'g1', name: 'grep', args: {path: '/指南.md', patterns: ['优先审查', '快速预审']}, status: 'running'});
  let view = toolPresentation(traces[0]);
  assert.equal(view.action, 'grep');
  assert.equal(view.target, '优先审查 / 快速预审');
  assert.equal(view.note, '');
  upsertTool(traces, {tool_call_id: 'g1', status: 'success', summary: {match_count: 10, truncated: true}});
  view = toolPresentation(traces[0]);
  assert.equal(view.note, '10 个命中行 · 结果已截断');
  const {activityLabel} = require('../../app/static/pilot_demo/app.js');
  assert.equal(activityLabel({tools: traces}), 'grep 1 次');
});

test('模型生成的非法 grep 参数不会导致轨迹页面崩溃', () => {
  const view = toolPresentation({name: 'grep', args: {patterns: '不是列表'}, status: 'error'});
  assert.equal(view.target, '不是列表');
  assert.equal(view.note, '调用失败');
});

test('重复开始事件不会创建第二行或覆盖完成状态', () => {
  const traces = [];
  const started = { tool_call_id: 'r1', name: 'read', args: { path: '/目录/指南.md' }, status: 'running' };
  upsertTool(traces, started);
  upsertTool(traces, { tool_call_id: 'r1', status: 'success', summary: { start_line: 181, end_line: 245 } });
  upsertTool(traces, started);
  assert.equal(traces.length, 1);
  assert.equal(traces[0].status, 'success');
  assert.equal(toolPresentation(traces[0]).target, '指南.md');
  assert.equal(toolPresentation(traces[0]).location, 'L181–245');
});

test('最终轨迹补齐实时缺失项，并以服务端状态为准', () => {
  const live = [{ tool_call_id: 's1', name: 'search', status: 'running', summary: { hit_count: 5 } }];
  const final = [{ tool_call_id: 's1', name: 'search', status: 'success' }, { tool_call_id: 'r1', name: 'read', status: 'error' }];
  const merged = reconcileTools(live, final);
  assert.equal(merged.length, 2);
  assert.equal(merged[0].status, 'success');
  assert.equal(merged[0].summary.hit_count, 5);
  assert.equal(merged[1].status, 'error');
  assert.equal(reconcileTools([], final).length, 2);
});

function streamOf(events) {
  const bytes = new TextEncoder().encode(events.map(JSON.stringify).join('\n') + '\n');
  return new ReadableStream({ start(controller) {
    // 按单字节拆分，包括中文 UTF-8 与行边界。
    for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
    controller.close();
  } });
}

test('UTF-8 分片按真实顺序交付，并独立检测本轮完成', async () => {
  const events = [
    { event: 'tool_started', data: { tool_call_id: 'r1', name: 'read' } },
    { event: 'tool_completed', data: { tool_call_id: 'r1', summary: { path: '/指南.md' } } },
    { event: 'completed', data: { answer: '中文回答' } },
  ];
  const received = [];
  await consumeAgentStream(streamOf(events), event => received.push(event));
  assert.deepEqual(received, events);
  // 前一次成功不能掩盖第二次没有 completed 的连接。
  await assert.rejects(consumeAgentStream(streamOf(events.slice(0, 2)), () => {}), /未收到最终回答/);
});

test('流中失败保持之前事件，不能渲染成成功', async () => {
  const received = [];
  await assert.rejects(consumeAgentStream(streamOf([
    { event: 'tool_started', data: { tool_call_id: 'r1' } },
    { event: 'error', data: { message: '连接中断' } },
  ]), event => received.push(event)), /连接中断/);
  assert.equal(received.length, 1);
});

test('未知的 read 返回范围不能伪装成实际读取行数', () => {
  const view = toolPresentation({ name: 'read', args: { path: '/目录/指南.md', start_line: 181, limit: 80 }, status: 'running' });
  assert.equal(view.location, '从 L181 开始');
  assert.equal(view.note, '');
});
