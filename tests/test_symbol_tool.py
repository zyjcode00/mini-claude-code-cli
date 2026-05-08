# tests/test_symbol_tool.py
# 测试符号工具功能
import sys
sys.path.insert(0, '.')

from tools.symbol_tool import ListSymbolsTool, FindSymbolTool, SymbolExtractor


def test_list_all_symbols():
    """测试列出所有符号功能"""
    tool = ListSymbolsTool()
    result = tool.run(path='.', symbol_type='all')
    
    # 应该找到符号
    assert '📊 项目符号索引' in result or '未找到任何符号定义' in result
    
    # 如果找到符号，应该包含类型标识
    if '📊 项目符号索引' in result:
        assert '📦' in result or '⚡' in result


def test_list_class_symbols():
    """测试只列出类定义"""
    tool = ListSymbolsTool()
    result = tool.run(path='.', symbol_type='class')
    
    # 应该找到类定义
    assert '📊 项目符号索引' in result or '未找到任何符号定义' in result
    
    # 如果找到，应该只有类（不包含⚡函数标识）
    if '📊 项目符号索引' in result:
        # 检查是否包含 BaseTool 等已知类
        assert 'BaseTool' in result or 'SymbolInfo' in result


def test_find_symbol_by_name():
    """测试根据名称查找符号"""
    tool = FindSymbolTool()
    result = tool.run(name='BaseTool', path='.')
    
    # 应该找到 BaseTool 类
    assert '🔍 找到' in result
    assert 'BaseTool' in result
    assert 'CLASS' in result
    assert '📁 文件:' in result
    assert '📍 行号:' in result


def test_find_symbol_partial_match():
    """测试部分匹配查找"""
    tool = FindSymbolTool()
    result = tool.run(name='Tool', path='.')
    
    # 应该找到多个 Tool 类
    assert '🔍 找到' in result
    # 至少应该找到 BaseTool, SearchTool 等
    assert 'Tool' in result


def test_find_nonexistent_symbol():
    """测试查找不存在的符号"""
    tool = FindSymbolTool()
    result = tool.run(name='NonexistentSymbol12345', path='.')
    
    # 应该返回未找到
    assert '未找到符号' in result


def test_symbol_extractor():
    """测试符号提取器核心功能"""
    extractor = SymbolExtractor('.')
    symbols = extractor.extract_from_project()
    
    # 应该找到符号
    assert len(symbols) > 0
    
    # 检查符号属性
    for symbol in symbols:
        assert symbol.name
        assert symbol.type in ['class', 'function']
        assert symbol.file
        assert symbol.start_line > 0
        assert symbol.end_line >= symbol.start_line


def test_symbol_extractor_find():
    """测试符号提取器的查找功能"""
    extractor = SymbolExtractor('.')
    extractor.extract_from_project()
    
    # 查找 BaseTool
    matches = extractor.find_symbol('BaseTool')
    assert len(matches) > 0
    assert any(s.name == 'BaseTool' for s in matches)


def test_symbol_info_model():
    """测试符号信息模型"""
    from tools.symbol_tool import SymbolInfo
    
    symbol = SymbolInfo(
        name='TestClass',
        type='class',
        file='test.py',
        start_line=10,
        end_line=20,
        docstring='Test docstring'
    )
    
    assert symbol.name == 'TestClass'
    assert symbol.type == 'class'
    assert symbol.file == 'test.py'
    assert symbol.start_line == 10
    assert symbol.end_line == 20
    assert symbol.docstring == 'Test docstring'


def test_code_snippet_display():
    """测试代码片段显示"""
    tool = FindSymbolTool()
    result = tool.run(name='BaseTool', path='.')
    
    # 应该包含代码片段
    if '🔍 找到' in result:
        assert '📝 代码片段' in result or 'FILE' in result


if __name__ == "__main__":
    # 直接运行时执行测试
    test_list_all_symbols()
    test_list_class_symbols()
    test_find_symbol_by_name()
    test_find_symbol_partial_match()
    test_find_nonexistent_symbol()
    test_symbol_extractor()
    test_symbol_extractor_find()
    test_symbol_info_model()
    test_code_snippet_display()
    print("✅ 所有测试通过！")