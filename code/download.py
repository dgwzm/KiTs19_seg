from pathlib import Path
import shutil
import os
import sys
import time
import paramiko
import requests

def get_request():
    header={
        'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1'
    }
    txt_path=r"D:\BaiduYunDownload\KiTs19_seg\code\file_name.txt"
    f=open(txt_path)
    d=f.readlines()[0]
    temp_f = Path(__file__).parent / "temp.tmp"
    new_file=r"D:\BaiduYunDownload\KiTs_seg_data\master_00000.nii.gz"
    with requests.get(d, headers=header) as r:
        with temp_f.open('wb') as f:
            shutil.copyfileobj(r.raw, f)
    shutil.move(str(temp_f), new_file)

    print(d)
    print("add pro!!")

if __name__ == "__main__":
    cmd = "ssh -p 22099 linda@5pi0081444.zicp.vip"
    returned_value = os.system(cmd)  # returns the exit code in unix
    print('returned value:', returned_value)
    trans = paramiko.Transport(('5pi0081444.zicp.vip', 22099))
    # 建立连接
    trans.connect(username='linda', password='ore$2020')

    # 实例化一个 sftp对象,指定连接的通道
    sftp = paramiko.SFTPClient.from_transport(trans)
    # 发送文件
    sftp.put(localpath='/tmp/11.txt', remotepath='/tmp/22.txt')
    # 下载文件

    trans.close()
