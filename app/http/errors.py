'''FastAPI接口统一业务异常'''
from fastapi import Request
from fastapi.responses import JSONResponse

class ApiError(Exception):
    '''可以转换为统一的HTTP错误响应业务异常'''
    def __init__(
            self,
            *,
            status_code:int,
            code:str,
            message:str,
    )->None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
async def api_error_handler(
        request:Request,
        error:ApiError,
)->JSONResponse:
    '''将 ApiError 转换为统一的 JSON 响应'''
    return JSONResponse(
        status_code = error.status_code,
        content={
            'error':{
                'code':error.code,
                'message':error.message,
            }
        }
    )
