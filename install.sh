#!/bin/bash

# ==========================================
# 环境配置变量
# ==========================================
ENV_NAME="verl-agent"
PYTHON_VERSION="3.12"
TENCENT_MIRROR="-i https://mirrors.cloud.tencent.com/pypi/simple/"

echo "🚀 开始环境装配流程..."

# ==========================================
# 1. Conda 环境配置与激活
# ==========================================
# 检查环境是否存在
if conda info --envs | grep -E -q "^${ENV_NAME}\s"; then
    echo "✅ [跳过] Conda 环境 '${ENV_NAME}' 已存在。"
else
    echo "🔄 [创建] 正在创建 Conda 环境 '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    if [ $? -ne 0 ]; then
        echo "❌ [失败] Conda 环境创建失败，请检查 Conda 配置。"
        exit 1
    fi
fi

# 在脚本中激活 Conda 需要先初始化 shell hook
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
echo "✅ 已激活环境: ${ENV_NAME}"

# ==========================================
# 2. 通用带 Fallback 机制的安装函数
# ==========================================
# 参数1: 模块名称说明
# 参数2: 验证命令 (执行成功表示已安装)
# 参数3: 安装命令
install_package() {
    local pkg_desc="$1"
    local check_cmd="$2"
    local install_cmd="$3"

    # 1. 检测是否已安装
    if eval "$check_cmd" >/dev/null 2>&1; then
        echo "✅ [跳过] $pkg_desc 已满足要求。"
        return 0
    fi

    echo "🔄 [安装] 开始安装 $pkg_desc ..."
    
    # 2. 尝试默认源安装
    if eval "$install_cmd"; then
        echo "✅ [成功] $pkg_desc 安装完成。"
    else
        echo "⚠️ [重试] 默认源安装失败，启动腾讯云镜像源重试..."
        # 3. 尝试腾讯云镜像源
        if eval "$install_cmd $TENCENT_MIRROR"; then
            echo "✅ [成功] 使用腾讯云源安装 $pkg_desc 完成。"
        else
            echo "❌ [失败] $pkg_desc 安装彻底失败，请排查网络或依赖冲突。"
            exit 1
        fi
    fi
}

# ==========================================
# 3. 执行安装任务
# ==========================================

# 任务 A: 安装 PyTorch 2.6.0 (GPU 版)
# 验证逻辑：版本包含 2.6.0 且 CUDA 可用
TORCH_CHECK="python -c \"import torch; exit(0) if '2.6.0' in torch.__version__ and torch.cuda.is_available() else exit(1)\""
TORCH_INSTALL="pip install torch==2.6.0"
install_package "PyTorch 2.6.0 (GPU)" "$TORCH_CHECK" "$TORCH_INSTALL"

# 任务 B: 安装 vLLM 0.11.0
VLLM_CHECK="python -c \"import vllm; exit(0) if '0.11.0' in vllm.__version__ else exit(1)\""
VLLM_INSTALL="pip install vllm==0.11.0"
install_package "vLLM 0.11.0" "$VLLM_CHECK" "$VLLM_INSTALL"

# 任务 C: 安装 flash-attn 2.7.4.post1
FLASH_CHECK="python -c \"import flash_attn; exit(0) if '2.7.4.post1' in flash_attn.__version__ else exit(1)\""
FLASH_INSTALL="pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir"
install_package "flash-attn 2.7.4.post1" "$FLASH_CHECK" "$FLASH_INSTALL"

# 任务 D: 安装本地项目可编辑模式
# 注意：对于本地目录，直接执行 `pip install -e .`，速度通常很快，遇到依赖缺失时同样会触发 fallback
echo "🔄 [安装] 正在安装当前项目 (pip install -e .) ..."
if pip install -e .; then
    echo "✅ [成功] 当前项目安装完成。"
else
    echo "⚠️ [重试] 默认源安装失败，使用腾讯云源重试..."
    if pip install -e . $TENCENT_MIRROR; then
        echo "✅ [成功] 当前项目安装完成 (腾讯云源)。"
    else
        echo "❌ [失败] 当前项目安装失败。"
        exit 1
    fi
fi

echo "🎉 所有环境装配步骤已顺利完成！"