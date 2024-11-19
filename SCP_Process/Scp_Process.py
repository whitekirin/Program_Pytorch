import paramiko
from scp import SCPClient
import os
from Load_process.file_processing import Process_File

class SCP():
    def __init__(self) -> None:
        pass
    def createSSHClient(self, server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        client.connect(server, port, user, password)

        return client
    
    def Process_Main(self, Remote_Save_Root, Local_Save_Root, File_Name):
        Process_File_Tool = Process_File()

        ssh = self.createSSHClient("10.1.29.28", 31931, "root", "whitekirin")

        Process_File_Tool.JudgeRoot_MakeDir(Local_Save_Root)

        with SCPClient(ssh.get_transport()) as scp:
            scp.get(Remote_Save_Root, Local_Save_Root + "/" + File_Name)

        os.remove(Remote_Save_Root + "/" + File_Name, Local_Save_Root)

        print("傳輸成功\n")