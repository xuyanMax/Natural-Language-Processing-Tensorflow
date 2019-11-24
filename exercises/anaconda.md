## Anaconda CLI

- `conda env list` 查看环境列表
- `conda create -n envName python=3` #create the latest python 3 environment
- `conda activate envName` 切换到指定环境
- `conda deactivate`  切换回原环境
- `conda list` 列出当前环境所有包
- `conda remove -n envName --all` 删除指定环境
- `conda install requests` 在当前环境下, 安装包
- `conda remove requests` 在当前环境下, 删除包
- `conda env export > environment.yaml` 将当前环境的包名及版本号导出
- `conda env create -f environment.yaml` 利用配置文件创建虚拟环境 