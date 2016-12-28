cd ..\
set SOLUTION_DIR=%cd%

set INCLUDE_PROTO_DIR=%SOLUTION_DIR%\include\proto
set SRC_PROTO_DIR=%SOLUTION_DIR%\src\proto
set PROTO_TEMP_DIR=%SRC_PROTO_DIR%\temp
set PROTO_DIR=%SOLUTION_DIR%\packages\protoc_x64.2.6.1\build\native

mkdir "%PROTO_TEMP_DIR%"

"%PROTO_DIR%\protoc" --proto_path="%SRC_PROTO_DIR%" --cpp_out="%PROTO_TEMP_DIR%" "%SRC_PROTO_DIR%\surfing.proto"

mkdir "%INCLUDE_PROTO_DIR%"

fc /b "%PROTO_TEMP_DIR%\surfing.pb.h" "%INCLUDE_PROTO_DIR%\surfing.pb.h" > NUL

if errorlevel 1 (
    move /y "%PROTO_TEMP_DIR%\surfing.pb.h" "%INCLUDE_PROTO_DIR%\surfing.pb.h"
    move /y "%PROTO_TEMP_DIR%\surfing.pb.cc" "%SRC_PROTO_DIR%\surfing.pb.cc"
)


#rmdir /S /Q "PROTO_TEMP_DIR"

pause