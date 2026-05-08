# core/session_manager.py
"""会话管理器，用于列出历史会话、查找未完成会话等"""
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class SessionManager:
    def __init__(self, sessions_dir: str = None):
        if sessions_dir is None:
            # 默认使用项目根目录下的 sessions 文件夹
            project_root = Path(__file__).parent.parent
            sessions_dir = project_root / 'sessions'
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def list_sessions(self) -> List[Dict]:
        """列出所有历史会话，按修改时间倒序排列"""
        sessions = []
        for file in self.sessions_dir.glob('*.json'):
            try:
                stat = file.stat()
                sessions.append({
                    'filename': file.name,
                    'path': str(file),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime),
                    'size': stat.st_size
                })
            except Exception as e:
                print(f"Warning: Could not read session file {file}: {e}")
        
        # 按修改时间倒序排列
        sessions.sort(key=lambda x: x['modified_time'], reverse=True)
        return sessions
    
    def get_session_info(self, filename: str) -> Optional[Dict]:
        """获取指定会话的详细信息"""
        filepath = self.sessions_dir / filename
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否有未完成的任务
            plan_data = data.get('plan', {})
            tasks = plan_data.get('tasks', [])
            has_incomplete = any(t.get('status') != 'done' for t in tasks)
            
            return {
                'filename': filename,
                'path': str(filepath),
                'has_incomplete_tasks': has_incomplete,
                'goal': plan_data.get('current_goal', ''),
                'task_count': len(tasks),
                'completed_count': sum(1 for t in tasks if t.get('status') == 'done'),
                'summary': data.get('history_summary', '')[:200] + '...' if data.get('history_summary') else ''
            }
        except Exception as e:
            print(f"Warning: Could not parse session file {filepath}: {e}")
            return None
    
    def find_latest_incomplete_session(self) -> Optional[Dict]:
        """查找最近的未完成会话"""
        sessions = self.list_sessions()
        for session in sessions:
            info = self.get_session_info(session['filename'])
            if info and info['has_incomplete_tasks']:
                return info
        return None
    
    def format_session_list(self) -> str:
        """格式化会话列表用于显示"""
        sessions = self.list_sessions()
        if not sessions:
            return "暂无历史会话。"
        
        lines = ["\n📋 历史会话列表："]
        lines.append("-" * 60)
        
        for i, session in enumerate(sessions, 1):
            info = self.get_session_info(session['filename'])
            if info:
                status = "⏳ 未完成" if info['has_incomplete_tasks'] else "✅ 已完成"
                time_str = session['modified_time'].strftime("%m-%d %H:%M")
                goal = info['goal'][:30] + "..." if len(info['goal']) > 30 else info['goal']
                lines.append(f"{i}. [{status}] {session['filename']} ({time_str})")
                if goal:
                    lines.append(f"   目标: {goal}")
                lines.append(f"   进度: {info['completed_count']}/{info['task_count']} 任务")
        
        return "\n".join(lines)  
