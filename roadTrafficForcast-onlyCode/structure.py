'''
打印项目结构
'''
import os
import sys
def tree(directory: str, padding: str, level: int, print_files: bool = False):
    if not os.path.isdir(directory) or level < 0:
        return

    files = os.listdir(directory)

    # sort directories first, then files
    files.sort(key=lambda f: os.path.isdir(os.path.join(directory, f)))

    for file in files:
        full_path = os.path.join(directory, file)

        if os.path.isdir(full_path):
            print(padding[:-1] + '+--' + file + '/')
            new_padding = padding + ' '
            tree(full_path, new_padding, level-1, print_files)
        elif print_files:
            print(padding[:-1] + '+--' + file)


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.dirname(__file__))
    print(project_root)
    tree(project_root, '', 3, True)
