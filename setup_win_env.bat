@echo off
chcp 65001 >nul
echo ====================================================
echo   CoMaPOI Windows 本地 Conda 环境一键部署脚本
echo ====================================================
echo.

:: Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 系统未检测到 conda 命令，请确保已安装 Anaconda/Miniconda 并将 conda 目录加入系统环境变量 PATH。
    pause
    exit /b 1
)

echo [1/3] 正在创建 Conda 环境: comapoi_win (Python 3.10) ...
call conda create -n comapoi_win python=3.10 -y

echo.
echo [2/3] 正在激活环境并配置 pip 国内镜像源 ...
:: Force activation
call conda activate comapoi_win
:: Use aliyun pip mirror for faster download in China
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

echo.
echo [3/3] 正在安装 Windows 本地依赖库 (requirements_win.txt) ...
pip install -r requirements_win.txt

echo.
echo ====================================================
echo   环境部署完成！
echo ====================================================
echo.
echo 激活环境的命令:
echo   conda activate comapoi_win
echo.
echo 在本地 Windows 上使用百炼 API 进行前向推理 (Forward) 冒烟测试示例命令：
echo   python inference_forward_new.py --dataset ca --num_samples 5 --batch_size 1 --candidate_fusion_strategy rrf --use_hsid --agent1_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent1_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent1_api "qwen-plus" --agent2_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent2_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent2_api "qwen-plus" --agent3_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent3_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent3_api "qwen-plus"
echo.
pause
