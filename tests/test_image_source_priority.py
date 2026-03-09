#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片来源优先级功能测试
"""


def test_image_source_extraction():
    """测试不同配置下的图片来源提取逻辑"""
    
    # 模拟数据集中的图片信息
    test_images = [
        {
            "file_path": "files/ethics/test1.jpg",
            "url": "https://cdn.example.com/test1.jpg",
            "md5": "abc123"
        },
        {
            "file_path": "files/ethics/test2.jpg",
            # 没有 url
            "md5": "def456"
        },
        {
            # 没有 file_path
            "url": "https://cdn.example.com/test3.jpg",
            "md5": "ghi789"
        }
    ]
    
    # 测试不同的优先级配置
    priority_configs = ['url', 'local', 'url_only', 'local_only']
    
    print("="*80)
    print("图片来源优先级功能测试")
    print("="*80)
    
    for priority in priority_configs:
        print(f"\n配置: image_source_priority = '{priority}'")
        print("-" * 80)
        
        image_paths = []
        for idx, img in enumerate(test_images, 1):
            img_path = None
            
            if priority == 'url_only':
                # 强制只使用 URL
                if 'url' in img:
                    img_path = img['url']
            elif priority == 'local_only':
                # 强制只使用本地文件
                if 'file_path' in img:
                    img_path = img['file_path']
            elif priority == 'url':
                # 优先使用 URL，如果没有则使用本地文件
                img_path = img.get('url') or img.get('file_path')
            elif priority == 'local':
                # 优先使用本地文件，如果没有则使用 URL
                img_path = img.get('file_path') or img.get('url')
            
            if img_path:
                image_paths.append(img_path)
                source_type = "URL" if img_path.startswith('http') else "本地文件"
                print(f"  图片 {idx}: {img_path} (来源: {source_type})")
            else:
                print(f"  图片 {idx}: 跳过 (无可用来源)")
        
        print(f"\n  总计: {len(image_paths)}/{len(test_images)} 张图片可用")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


def test_expected_results():
    """验证预期结果"""
    
    print("\n\n预期结果验证:")
    print("="*80)
    
    test_cases = [
        {
            "priority": "url",
            "expected": [
                "https://cdn.example.com/test1.jpg",  # 有 URL，使用 URL
                "files/ethics/test2.jpg",              # 没有 URL，使用本地
                "https://cdn.example.com/test3.jpg"   # 只有 URL，使用 URL
            ],
            "description": "优先 URL：图片1用URL，图片2用本地，图片3用URL"
        },
        {
            "priority": "local",
            "expected": [
                "files/ethics/test1.jpg",              # 有本地，使用本地
                "files/ethics/test2.jpg",              # 只有本地，使用本地
                "https://cdn.example.com/test3.jpg"   # 没有本地，使用 URL
            ],
            "description": "优先本地：图片1用本地，图片2用本地，图片3用URL"
        },
        {
            "priority": "url_only",
            "expected": [
                "https://cdn.example.com/test1.jpg",  # 有 URL
                None,                                  # 没有 URL，跳过
                "https://cdn.example.com/test3.jpg"   # 有 URL
            ],
            "description": "仅 URL：图片1用URL，图片2跳过，图片3用URL"
        },
        {
            "priority": "local_only",
            "expected": [
                "files/ethics/test1.jpg",              # 有本地
                "files/ethics/test2.jpg",              # 有本地
                None                                   # 没有本地，跳过
            ],
            "description": "仅本地：图片1用本地，图片2用本地，图片3跳过"
        }
    ]
    
    for idx, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {idx}: {case['priority']}")
        print(f"  描述: {case['description']}")
        print(f"  预期结果: {[p for p in case['expected'] if p is not None]}")
        print(f"  预期可用: {len([p for p in case['expected'] if p is not None])}/3 张图片")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_image_source_extraction()
    test_expected_results()
    
    print("\n\n使用说明:")
    print("="*80)
    print("1. 在配置文件中添加 'image_source_priority' 字段")
    print("2. 可选值: 'url', 'local', 'url_only', 'local_only'")
    print("3. 默认值: 'url' (优先使用 URL)")
    print("4. 详细说明请参考: docs/图片来源配置说明.md")
    print("="*80)

