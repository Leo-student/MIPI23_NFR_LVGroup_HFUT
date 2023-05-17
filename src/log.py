#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import time
import os


class Log(object):
    '''
        封装后的logging
    '''
    instance = None
    
    
    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Log, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, logger=None, log_cate='search', ):
        '''
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        '''
        
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        # 创建一个handler，用于写入日志文件
        self.log_time = time.strftime("%Y_%m_%d")
         # 定义handler的输出格式
        self.formatter = logging.Formatter(
            '[%(asctime)s]->[%(levelname)s]%(message)s')
            # '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
            
       
        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

       
        
        ch.setFormatter(self.formatter)

        # 给logger添加handler
        
        self.logger.addHandler(ch)

        #  添加下面一句，在记录日志之后移除句柄
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)
        # 关闭打开的文件
        
        ch.close()
        
        

    def getlog(self):
        return self.logger
    
    def writelog(self, opt):
        # file_dir = os.getcwd() + '/../log/' + self.log_time
        file_dir = os.path.join(opt.results_dir, opt.experiment,'log')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        self.log_path = file_dir
        # self.log_name = self.log_path + "/"  + self.log_time + '.log'
        self.log_name = self.log_path + "/"  + opt.experiment + '.log'
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 这个是python3的
        fh.setLevel(logging.INFO)
        
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)
        fh.close()
        return self.logger