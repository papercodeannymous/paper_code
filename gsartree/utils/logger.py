"""
日志系统 - 结构化的日志记录工具
支持文件日志和控制台输出，便于训练过程追踪
"""
import os
import logging
from datetime import datetime
from typing import Optional


class Logger:
    """
    结构化日志器
    
    提供统一的日志接口，支持：
    - 控制台输出（彩色）
    - 文件持久化
    - 不同日志级别
    - 自动时间戳
    
    Attributes:
        logger: Python logging.Logger 实例
        log_dir: 日志目录
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        初始化日志器
        
        Args:
            log_dir: 日志文件保存目录
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        self.log_dir = log_dir
        self.logger = logging.getLogger(f"GSAR-Tree-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.logger.setLevel(log_level)
        
        # 避免重复添加 handler
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台 Handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件 Handler
        if file_output:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file = os.path.join(log_dir, f"gsartree_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file_path = log_file
    
    def debug(self, message: str):
        """调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """普通信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告信息"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """错误信息"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """严重错误"""
        self.logger.critical(message)
    
    def section(self, title: str):
        """打印分节标题"""
        separator = "=" * 80
        self.logger.info(f"\n{separator}")
        self.logger.info(f"{title.center(80)}")
        self.logger.info(f"{separator}\n")
    
    def subsection(self, title: str):
        """打印子节标题"""
        separator = "-" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{separator}")
    
    def metric(self, name: str, value: float, unit: str = ""):
        """记录指标"""
        self.logger.info(f"  {name}: {value:.4f}{unit}")
    
    def success(self, message: str):
        """成功信息"""
        self.logger.info(f"✓ {message}")
    
    def failure(self, message: str):
        """失败信息"""
        self.logger.error(f"✗ {message}")
    
    def progress(self, current: int, total: int, prefix: str = "Progress"):
        """记录进度"""
        percentage = (current / total * 100) if total > 0 else 0
        self.logger.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")
    
    def get_log_file_path(self) -> Optional[str]:
        """获取日志文件路径"""
        return getattr(self, 'log_file_path', None)


# 全局默认日志器实例
_default_logger: Optional[Logger] = None


def get_logger(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True
) -> Logger:
    """
    获取或创建全局日志器
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
    
    Returns:
        Logger 实例
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = Logger(log_dir, log_level, console_output, file_output)
    
    return _default_logger


def reset_logger():
    """重置全局日志器"""
    global _default_logger
    _default_logger = None
