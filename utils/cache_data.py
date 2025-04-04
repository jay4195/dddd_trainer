import json
import os
import random

import tqdm

from configs import Config
from loguru import logger


class CacheData:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        if os.path.exists(self.project_path):
            self.cache_path = os.path.join(self.project_path, "cache")
        else:
            logger.error("Project {} is not exists!".format(project_name))
            exit()
        self.config = Config(project_name)
        self.conf = self.config.load_config()
        self.bath_path = self.conf['System']['Path']
        self.allow_ext = []

    def cache(self, base_path: str, search_type="name"):
        self.bath_path = base_path
        self.allow_ext = self.conf["System"]["Allow_Ext"]
        self.__get_label_from_name(base_path=base_path)

    def __find_files_os_walk(self, start_dir):
        """
        使用 os.walk 递归查找文件并返回相对路径列表。
        Args:
            start_dir (str): 要开始搜索的目录路径。
        Returns:
            list: 包含所有找到文件相对路径的字符串列表。
                  如果 start_dir 无效，则返回空列表。
        """
        relative_paths = []
        # 确保 start_dir 存在且是一个目录
        if not os.path.isdir(start_dir):
            print(f"错误: 目录 '{start_dir}' 不存在或不是一个有效的目录。")
            return relative_paths
        # os.walk 遍历目录树
        for dirpath, dirnames, filenames in os.walk(start_dir):
            for filename in filenames:
                # 构建文件的完整绝对路径
                full_path = os.path.join(dirpath, filename)
                # 计算相对于 start_dir 的路径
                relative_path = os.path.relpath(full_path, start_dir)
                relative_paths.append(relative_path)
        return relative_paths

    def __get_label_from_name(self, base_path: str):
        files = self.__find_files_os_walk(base_path)
        logger.info("\nFiles number is {}.".format(len(files)))
        self.__collect_data(files, base_path, [])

    def __extract_captcha_from_path(self, file_path: str) -> str | None:
        """
        从文件路径中提取验证码。

        验证码被定义为位于路径最后一部分（文件名）中，
        且在第一个下划线字符 '_' 之前的部分。

        Args:
            file_path (str): 包含验证码的文件路径字符串。
                             例如：'autogen_2st\\Y\\YQYZ_dbf8... .png'

        Returns:
            str | None: 提取到的验证码字符串。
                        - 如果成功提取，返回验证码（如 'YQYZ'）。
                        - 如果文件名以下划线开头，返回空字符串 ''。
                        - 如果文件名中没有下划线，返回整个文件名。
                        - 如果输入路径无效或无法获取文件名，返回 None。
        """
        if not file_path:
            # 处理空输入路径
            return None

        # 1. 获取路径中的最后一部分（文件名）
        #    os.path.basename 能正确处理 Windows (\) 和 POSIX (/) 分隔符
        filename = os.path.basename(file_path)

        # 2. 检查是否成功获取了文件名
        if not filename:
            # 如果路径以分隔符结尾（如 'some/dir/'），basename 可能返回空字符串
            return None

        # 3. 使用下划线分割文件名，最多分割一次
        parts = filename.split('_', 1)

        # 4. 返回分割后的第一部分。
        #    因为 split 总是返回至少包含一个元素的列表（除非输入为空，但已处理），
        #    所以可以直接安全地访问 parts[0]。
        return parts[0]

    def __collect_data(self, lines, base_path, error_files):
        labels = []
        caches = []

        for file in tqdm.tqdm(lines):
            filename = file
            label = self.__extract_captcha_from_path(file_path=file)
            if filename in error_files:
                continue
            label = label.replace(" ", "")
            if filename.split('.')[-1] in self.allow_ext:
                if " " in filename:
                    logger.warning("The {} has black. We will remove it!".format(filename))
                    continue
                caches.append('\t'.join([filename, label]))
                if not self.conf['Model']['Word']:
                    label = list(label)
                    labels.extend(label)
                else:
                    labels.append(label)

            else:
                logger.warning("\nFile({}) has a suffix that is not allowed! We will remove it!".format(file))
        labels = list(set(labels))
        if not self.conf['Model']['Word']:
            labels.insert(0, " ")
        logger.info("\nCollected labels is {}".format(json.dumps(labels, ensure_ascii=False)))
        self.conf['System']['Path'] = base_path
        self.conf['Model']['CharSet'] = labels
        self.config.make_config(config_dict=self.conf, single=self.conf['Model']['Word'])
        logger.info("\nWriting Cache Data!")
        del lines
        logger.info("\nCache Data Number is {}".format(len(caches)))
        logger.info("\nWriting Train and Val File.".format(len(caches)))
        val = self.conf['System']['Val']
        if 0 < val < 1:
            val_num = int(len(caches) * val)
        elif 1 < val < len(caches):
            val_num = int(val)
        else:
            logger.error("val setting vaild!")
            exit()
        random.shuffle(caches)
        train_set = caches[val_num:]
        val_set = caches[:val_num]
        del caches
        with open(os.path.join(self.cache_path, "cache.train.tmp"), 'w', encoding="utf-8") as f:
            f.write("\n".join(train_set))
        with open(os.path.join(self.cache_path, "cache.val.tmp"), 'w', encoding="utf-8") as f:
            f.write("\n".join(val_set))
        logger.info("\nTrain Data Number is {}".format(len(train_set)))
        logger.info("\nVal Data Number is {}".format(len(val_set)))
