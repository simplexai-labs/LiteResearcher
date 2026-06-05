这是一个前端用于可视化json形式的轨迹

启动方式，执行如下命令，这行命令的作用是将当前路径下的文件转发到对应的端口

```bash
python3 -m http.server 8123
```

随后访问http://localhost:8123/trajectory_visualizer.html

将任意jsonl文件加载到这个前端即可完成可视化