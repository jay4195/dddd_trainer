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

    def __get_label_from_name(self, base_path: str):
        files = self.__get_relative_files_recursive(base_path)
        logger.info("\nFiles number is {}.".format(len(files)))
        self.__collect_data(files, base_path, [])

    def __get_relative_files_recursive(self, directory):
        """
        递归获取指定目录下所有文件（包括子目录中的文件）的相对路径。

        Args:
            directory: 要查找的目录的绝对或相对路径。

        Returns:
            一个包含相对文件路径的列表。
        """
        relative_paths = []
        for root, _, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(filepath, directory)
                relative_paths.append(relative_path)
        return relative_paths

    def __extract_label_before_underscore(self, file_path):
        """
        从文件路径中提取最后一个路径分隔符后、下划线前的数字加字母组合。
        如果没有路径分隔符，则提取从字符串首字母开始到下划线前的内容。

        Args:
            file_path: 文件路径字符串。

        Returns:
            提取到的字符串，如果找不到下划线则返回整个文件名部分，
            如果路径为空或只包含分隔符则返回空字符串。
        """
        if not file_path:
            return ""

        last_separator_index = file_path.rfind(os.sep)

        if last_separator_index != -1:
            # 存在路径分隔符，取最后一个分隔符之后的部分作为文件名
            file_name = file_path[last_separator_index + 1:]
        else:
            # 不存在路径分隔符，整个路径作为文件名
            file_name = file_path

        underscore_index = file_name.find('_')

        if underscore_index != -1:
            return file_name[:underscore_index]
        else:
            return file_name

    def __collect_data(self, lines, base_path, error_files):
        labels = []
        caches = []

        for file in tqdm.tqdm(lines):
            filename = file
            label = self.__extract_label_before_underscore(file_path=filename)
            # print(f'{filename}\t{label}')
            if filename in error_files:
                continue
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
        logger.info("\nCollected labels are {}".format(json.dumps(labels, ensure_ascii=False)))
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
