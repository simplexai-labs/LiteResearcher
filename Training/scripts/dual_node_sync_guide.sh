#!/bin/bash
# 双节点Checkpoint同步 - 快速启动指南

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================"
echo "  🚀 双节点Checkpoint同步启动指南"
echo "========================================${NC}"
echo ""

# 检测当前节点
HOSTNAME=$(hostname)
NODE_IP=$(hostname -I | awk '{print $1}')

echo -e "${BLUE}当前节点信息:${NC}"
echo "  主机名: $HOSTNAME"
echo "  IP地址: $NODE_IP"
echo ""

# 节点识别
if [[ $NODE_IP == 172.27.72.103 ]]; then
    NODE_NAME="节点1 (Master)"
    OTHER_NODE="172.27.244.90 (Worker)"
elif [[ $NODE_IP == 172.27.244.90 ]]; then
    NODE_NAME="节点2 (Worker)"
    OTHER_NODE="172.27.72.103 (Master)"
else
    NODE_NAME="未知节点"
    OTHER_NODE="请手动配置"
fi

echo -e "${GREEN}节点角色: $NODE_NAME${NC}"
echo "  对端节点: $OTHER_NODE"
echo ""

echo "========================================"
echo "  📋 操作步骤"
echo "========================================"
echo ""

cat << 'EOF'
### 在每个节点上执行以下步骤：

#### 1️⃣  创建tmux会话
```bash
tmux new -s checkpoint_sync
```

#### 2️⃣  进入项目目录
```bash
cd /share/project/wanli/Search_Agent/verl
```

#### 3️⃣  启动同步守护进程 (终端实时显示)
```bash
bash scripts/checkpoint_sync_daemon.sh "qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl" 180
```

参数说明:
  - 第1个参数: 实验名称
  - 第2个参数: 检查间隔(秒), 推荐180-300秒

#### 4️⃣  Detach tmux (保持后台运行)
```
按键: Ctrl+B, 然后按 D
```

#### 5️⃣  重新连接到tmux
```bash
tmux attach -t checkpoint_sync
```

#### 6️⃣  查看所有tmux会话
```bash
tmux ls
```

#### 7️⃣  停止守护进程
```
在tmux会话中按: Ctrl+C
```

EOF

echo ""
echo "========================================"
echo "  🔧 当前节点快速启动命令"
echo "========================================"
echo ""
echo -e "${YELLOW}# 在tmux中运行此命令:${NC}"
echo "cd /share/project/wanli/Search_Agent/verl && \\"
echo "bash scripts/checkpoint_sync_daemon.sh \"qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl\" 180"
echo ""

echo "========================================"
echo "  📊 监控命令"
echo "========================================"
echo ""
cat << 'EOF'
# 查看日志文件
tail -f logs/checkpoint_sync.log

# 查看进程状态
ps aux | grep checkpoint_sync_daemon

# 查看已迁移的checkpoint
cat logs/migrated_checkpoints.txt

# 查看共享存储中的checkpoint
ls -lh /share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl/
EOF

echo ""
echo "========================================"
echo "  💡 提示"
echo "========================================"
echo ""
echo "1. 两个节点都要启动守护进程"
echo "2. 使用tmux可以保持进程在后台运行"
echo "3. 脚本会自动检测并避免重复迁移"
echo "4. 迁移到共享存储后,两个节点都能看到"
echo "5. 脚本会等待checkpoint完全写入后再迁移"
echo ""
echo "========================================"
