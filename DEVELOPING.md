# 开发与发布

本项目的首要兼容目标是：现有 PyAutoGUI 调用代码无需修改。Rust 快速路径属于内部实现，失败时由 Python/Win32 兼容层处理。

## API 兼容规则

- 保留既有公共函数、常量和别名，例如 `move`/`moveRel`、`write`/`typewrite`。
- 保留原参数名称、位置参数顺序与默认行为；新增能力使用可选关键字或独立对象。
- Python 层继续负责参数规范化、暂停、failsafe 和跨平台语义。
- Rust 优化不得要求调用方导入 `_rust_core`。
- 每次修改公共函数前更新 `tests/unit/test_api_contract.py`，先证明旧调用仍可绑定。

## 环境准备

Windows 建议使用 PowerShell 7、Python 3.12、稳定版 Rust、MSVC Build Tools 和 Windows SDK：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,benchmark]"
```

需要对 OpenCV/MSS 图像路径做性能测试时，再安装可选依赖：

```powershell
python -m pip install -e ".[dev,benchmark,vision]"
```

## 无副作用测试

默认测试目录是 `tests/unit`。`conftest.py` 会在导入 PyAutoGUI 前注入原生模块替身，再将平台适配器换成内存后端。因此测试期间不会移动鼠标、发送按键、滚轮输入、调整系统计时器或安装鼠标钩子。

```powershell
python -m pytest
python -m pytest --cov=pyautogui --cov-report=term-missing
python -m pytest tests/unit/test_windows_backend_fallback.py
```

`tests/unit/test_windows_backend_fallback.py` 模拟 User32 失败分支，只读取内存替身状态，不发送输入。上游遗留的 `tests/test_pyautogui.py` 和 `tests/test_automateboringstuff.py` 包含交互式及真实桌面操作，只用于人工兼容检查，不在默认收集范围。

完整本地门禁：

```powershell
python tools/check.py
```

只检查一侧：

```powershell
python tools/check.py --python-only
python tools/check.py --rust-only
```

## Rust 检查

```powershell
cargo fmt --all -- --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
```

Rust 单元测试应优先覆盖纯算法、坐标换算、输入批次编译和错误分支；系统 API 集成测试必须避免真实输入，并单独标记。

## 性能基准

```powershell
python -m pytest benchmarks --benchmark-only
```

参见 `benchmarks/README.md`。提交性能结论时记录 CPU、Windows 版本、Python/Rust 版本、电源计划、构建类型和样本 JSON。默认对比中位数，超过 10% 的变化需复核。

只读的实际桌面截图与找图基准：

```powershell
python tools/benchmark_native.py --iterations 30 --json .benchmarks/native.json
```

该工具只读取桌面像素，不发送键鼠事件。执行时应使用 Release Wheel 或 Release 扩展。

## 构建与校验

```powershell
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
python -m build
python tools/verify_dist.py dist
```

也可以执行整套流程：

```powershell
python tools/check.py --build
```

Windows wheel 使用 CPython 3.7 ABI3 标签，可由受支持的较新 CPython 复用。发布前还应在源码目录之外安装 wheel 并运行只读冒烟测试：

```powershell
$wheel = (Get-ChildItem dist\*.whl).FullName
python -m pip install --force-reinstall --no-deps $wheel
Push-Location $env:TEMP
python C:\path\to\project\tools\smoke_installed.py
Pop-Location
```

## 发布清单

1. 工作树干净，`Cargo.lock` 已提交。
2. `python tools/check.py --build` 全部通过。
3. wheel 名称包含 `cp37-abi3`，且只包含一个 `_rust_core` 原生扩展。
4. sdist 包含 Rust 源码、Cargo 锁文件、许可证和构建配置，不包含 `target/`。
5. 在源码目录外执行 `tools/smoke_installed.py`。
6. 记录基准结果并检查主要 API 契约。
