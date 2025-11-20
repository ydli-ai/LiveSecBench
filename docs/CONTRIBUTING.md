# 贡献指南

欢迎对 LiveSecBench 项目提出 issue、修复 bug 或提交新特性！我们感谢所有形式的贡献。

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [测试要求](#测试要求)
- [文档要求](#文档要求)
- [Pull Request 流程](#pull-request-流程)
- [许可与权利](#许可与权利)

## 行为准则

本项目遵循贡献者行为准则。参与项目时，请保持尊重、包容和专业的沟通态度。

## 如何贡献

### 报告 Bug

1. 在提交 issue 前，请先搜索是否已有相关问题
2. 使用清晰的标题和描述说明问题
3. 提供复现步骤、预期行为和实际行为
4. 包含环境信息（Python 版本、操作系统等）
5. 如可能，提供最小复现示例

### 提出新功能

1. 创建 issue 描述功能需求和使用场景
2. 讨论设计方案的可行性
3. 等待维护者确认后再开始实现

### 提交代码

1. Fork 本仓库
2. 创建功能分支（见下方开发流程）
3. 实现变更并添加测试
4. 确保所有测试通过
5. 提交 Pull Request

## 开发环境设置

### 1. Fork 和克隆仓库

```bash
# Fork 仓库后，克隆你的 fork
git clone https://github.com/YOUR_USERNAME/LiveSecBench.git
cd LiveSecBench
```

### 2. 设置上游仓库

```bash
git remote add upstream https://github.com/ydli-ai/LiveSecBench.git
```

### 3. 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n livesecbench python=3.10
conda activate livesecbench

# 或使用 venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

### 4. 安装依赖

```bash
# 安装项目（可编辑模式）
python -m pip install -e .

# 安装开发依赖
python -m pip install -e .[test]
```

### 5. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_config_manager.py -v

# 运行并显示覆盖率
pytest --cov=livesecbench --cov-report=html
```

## 代码规范

### Python 代码风格

- **格式化工具**：使用 `black` 格式化代码
  ```bash
  black livesecbench/ tests/
  ```

- **导入排序**：推荐使用 `isort` 排序 imports
  ```bash
  isort livesecbench/ tests/
  ```

- **类型注解**：为函数参数和返回值添加类型注解
  ```python
  def process_data(data: Dict[str, Any]) -> List[str]:
      ...
  ```

- **文档字符串**：为公共函数和类添加 docstring
  ```python
  def my_function(param: str) -> int:
      """
      函数功能描述
    
      args:
          param: 参数说明
    
      returns:
          返回值说明
      """
  ```

### 配置文件格式

- YAML 文件使用 2 空格缩进
- 保持配置项按逻辑分组
- 添加必要的注释说明

## 提交规范

### Commit Message 格式

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型（type）**：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具链相关

**示例**：
```
feat(scoring): add convergence detection for ELO rating

Add adaptive convergence detector to automatically stop
evaluation when scores stabilize.

Closes #123
```

### 分支命名

- `feat/feature-name`: 新功能
- `fix/bug-description`: Bug 修复
- `docs/documentation-update`: 文档更新
- `refactor/module-name`: 重构

## 测试要求

### 测试覆盖

- 新增功能必须包含单元测试
- 测试文件放在 `tests/` 目录下，与模块结构对应
- 测试函数命名：`test_<functionality>`

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_config_manager.py

# 运行并显示详细输出
pytest -v

# 运行并显示覆盖率
pytest --cov=livesecbench --cov-report=term-missing
```

### 测试最佳实践

- 使用 fixtures 共享测试数据
- 使用 mock 隔离外部依赖
- 测试边界条件和错误情况
- 保持测试独立，不依赖执行顺序

## 文档要求

### 代码文档

- 公共 API 必须有完整的 docstring
- 复杂逻辑添加行内注释
- 遵循 Google 或 NumPy 风格的 docstring

### 用户文档

- 用户可见的行为变更需更新 `docs/USER_GUIDE.md`
- API 变更需更新 `docs/API_DOCUMENTATION.md`
- 配置变更需更新配置示例和说明
- 重大变更需更新 `docs/CHANGELOG.md`

### README 更新

- 新功能或重要变更需同步更新 `README.md` 和 `README_EN.md`
- 保持中英文版本内容一致

## Pull Request 流程

### 1. 准备 PR

```bash
# 确保分支是最新的
git checkout main
git pull upstream main

# 创建功能分支
git checkout -b feat/my-feature

# 进行修改并提交
git add .
git commit -m "feat: add new feature"

# 推送到你的 fork
git push origin feat/my-feature
```

### 2. 提交 PR

1. 在 GitHub 上创建 Pull Request
2. 填写 PR 模板：
   - **标题**：清晰描述变更
   - **描述**：说明变更原因、影响范围
   - **关联 Issue**：如有，使用 `Closes #123` 格式
   - **测试**：说明测试覆盖情况
   - **检查清单**：确认已完成各项要求

### 3. PR 审查

- 维护者会审查代码并提出反馈
- 根据反馈进行修改并推送更新
- PR 通过审查后会被合并

### 4. PR 要求检查清单

在提交 PR 前，请确认：

- [ ] 代码遵循项目风格规范
- [ ] 所有测试通过
- [ ] 添加了必要的测试用例
- [ ] 更新了相关文档
- [ ] Commit message 遵循规范
- [ ] 没有引入破坏性变更（或已说明）
- [ ] 已同步上游 main 分支的最新代码

## 许可与权利

提交代码即表示同意该贡献遵循本项目的 Apache License 2.0 许可协议（参见仓库根目录的 `LICENSE` 文件）。

## 获取帮助

- **Issue**: 在 GitHub Issues 中提问
- **讨论**: 在 PR 中 @ 相关维护者
- **文档**: 查看 `docs/` 目录下的详细文档

感谢您对 LiveSecBench 项目的贡献！🎉