# ViT中归一化方法的分析与改进
tju 3021001526 张铨 
神经网络与深度学习课程设计

# 实验环境
torch2.1.2+cu12.1
pip install -r requirements.txt

# 数据集下载
根目录下创建文件夹data，若存在则无需创建，直接运行main.py会自动进行下载cifar10和cifar100数据集到本地data文件夹下。

# 运行方式
python main.py

# 实验结果
<h4>单元格跨行跨列:</h4>   <!--标题-->
<table border="1" width="500px" cellspacing="2">
<tr>
  <th align="center">Model</th>
  <th colspan="3" align="center">Cifar100</th>
  <th colspan="3" >Cifar10</th>
</tr>
<tr>
  <td>Model</td>
  <td>ViT</td>
  <td>T2T-ViT</td>
  <td>SwinV2</td>
  <td>ViT</td>
  <td>T2T-ViT</td>
  <td>SwinV2</td>
</tr>
<tr>
  <td>BN</td>
  <td>11.52</td>
  <td>-</td>
  <td>48.31</td>
  <td>-</td>
  <td>55.23</td>
  <td>72.15</td>
</tr>
<tr>
  <td>LN</td>
  <td>12.09</td>
  <td>-</td>
  <td>45.21</td>
  <td>-</td>
  <td>49.82</td>
  <td>71.12</td>
</tr>
<tr>
  <td>GN(8)</td>
  <td>9.46</td>
  <td>-</td>
  <td>45.79</td>
  <td>-</td>
  <td>57.20</td>
  <td>70.87</td>
</tr>
<tr>
  <td>MABN</td>
  <td>7.99</td>
  <td>-</td>
  <td>47.80</td>
  <td>-</td>
  <td>50.92</td>
  <td>70.42</td>
</tr>
<tr>
  <td>PN</td>
  <td>11.01</td>
  <td>-</td>
  <td>45.65</td>
  <td>-</td>
  <td>44.71</td>
  <td>70.24</td>
</tr>
<tr>
  <td>UN</td>
  <td>9.57</td>
  <td>-</td>
  <td>45.07</td>
  <td>-</td>
  <td>52.64</td>
  <td>68.14</td>
</tr>
</table>

