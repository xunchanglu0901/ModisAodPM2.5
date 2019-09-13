# MODIS04_3K_PM25
>Environmental Science: Predicting PM with MODIS data(AOD)  
>使用NASA MODIS Aqua/Terra 卫星的气溶胶L2级别数据, 即MOD04_3K, 来预测日均PM.

## 数据来源 DATA
>[气溶胶(MODIS04),](https://ladsweb.modaps.eosdis.nasa.gov/search/) 
>[污染物/空气质量,](http://beijingair.sinaapp.com/) 
>[气象数据,](https://darksky.net/dev) 
>[程序包.](https://www.lfd.uci.edu/~gohlke/pythonlibs/) 
>>再次感谢数据提供者!
## 气溶胶文件夹 Aerosol
#### [空间转换模块.py:](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B0%94%E6%BA%B6%E8%83%B6/%E7%A9%BA%E9%97%B4%E8%BD%AC%E6%8D%A2%E6%A8%A1%E5%9D%97.py)
>批量读取.hdf文件,针对各个监测站经纬度获取三个不同长度半径的同心圆,再按照角度划分为不同区域.
#### [四种插值方法和结果.py:](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B0%94%E6%BA%B6%E8%83%B6/%E5%9B%9B%E7%A7%8D%E6%8F%92%E5%80%BC%E6%96%B9%E6%B3%95%E5%92%8C%E7%BB%93%E6%9E%9C.py)
>批量;在时间上使用KNN和ewm方法进行插值; 空间上使用IDW和Iterative方法进行插值.
  
## 污染物文件夹 Pollution
#### [日均污染物数据整理.py:](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B1%A1%E6%9F%93%E7%89%A9/%E6%97%A5%E5%9D%87%E6%B1%A1%E6%9F%93%E7%89%A9%E6%95%B0%E6%8D%AE%E6%95%B4%E7%90%86.py)
>批量读取城市/站点空气质量数据; 可按监测站输出空气质量时间序列数据.
#### [四种插值方法和结果.py](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B1%A1%E6%9F%93%E7%89%A9/%E5%9B%9B%E7%A7%8D%E6%8F%92%E5%80%BC%E6%96%B9%E6%B3%95%E5%92%8C%E7%BB%93%E6%9E%9C.py)
>批量;在时间上使用KNN和ewm方法进行插值; 空间上使用IDW和Iterative方法进行插值.
#### [权重融合_熵权法.py:](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B1%A1%E6%9F%93%E7%89%A9/%E6%9D%83%E9%87%8D%E8%9E%8D%E5%90%88_%E7%86%B5%E6%9D%83%E6%B3%95.py)
>批量,按照熵权法对四种插值方法进行融合.
  
## 气象数据文件夹 DarkSky API
#### [DarkSkyAPI日均](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE/DarkSkyAPI%E6%97%A5%E5%9D%87.py)/[逐时.py: ](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE/DarkSkyAPI%E9%80%90%E6%97%B6.py)
>批量爬取对应经纬度的天气历史数据和天气预报数据.
#### [日均](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE/%E6%97%A5%E5%9D%87%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.py)/[逐时气象数据处理.py](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE/%E9%80%90%E6%97%B6%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.py), [逐时气象数据均值处理.py:](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE/%E9%80%90%E6%97%B6%E6%B0%94%E8%B1%A1%E6%95%B0%E6%8D%AE%E5%9D%87%E5%80%BC%E5%A4%84%E7%90%86.py)
>批量处理气象数据格式, 如时间范围删选, 时间格式调整, 逐时均值.

## 建模文件夹 Model
* 初步建立模型并应用交叉验证和重采样方法, 以下模型有待后续完善:  
> 深度学习:
>>[分布式特征感知神经网络](https://github.com/xunchanglu0901/MODIS_AOD_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E5%88%86%E5%B8%83%E5%BC%8F%E7%89%B9%E5%BE%81%E6%84%9F%E7%9F%A5%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.py)
> 神经网络:
>>[单输入模型](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E5%8D%95%E8%BE%93%E5%85%A5.py),
  [多输入模型 ](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E5%A4%9A%E8%BE%93%E5%85%A5.py); 
  [多输入模型+Boosting ](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E5%A4%9A%E8%BE%93%E5%85%A5%2Bboosting.py);  
>>[MPL神经网络模型+Bagging ](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%2BBagging.py); [MPL神经网络模型+KFold交叉验证 ](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%2B%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81.py);  
>>[MPL神经网络模型+KFold交叉验证+Boosting ](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%2B%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%2BBoosting.py); 
  [MPL神经网络模型+KFold交叉验证+Bagging](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%2B%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%2BBagging.py).  
  
>线性回归
 >>[线性回归模型+KFold交叉验证+Boosting ](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%2B%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%2BBoosting.py); 
  [线性回归模型+KFold交叉验证+Bagging](https://github.com/xunchanglu0901/MODIS04_3K_PM25/blob/master/%E5%BB%BA%E6%A8%A1/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%2B%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81%2BBagging.py).   
  
## 坐标文件夹 Longitude&Latitude
>[全国监测点: ](https://github.com/xunchanglu0901/MODIS_AOD_PM25/blob/master/%E5%9D%90%E6%A0%87/%E7%AB%99%E7%82%B9%E5%88%97%E8%A1%A8-2018.11.08%E8%B5%B7%20-%20%E5%89%AF%E6%9C%AC.xlsx)编码, 名称, 所属城市, 经纬度.

## 扩展库和扩展程序
#### [库](https://github.com/xunchanglu0901/MODIS04_3K_PM25/tree/master/%E6%89%A9%E5%B1%95%E5%BA%93%E5%92%8C%E6%89%A9%E5%B1%95%E7%A8%8B%E5%BA%8F/%E5%BA%93)
>气象数据: DarkSkyAPI, 数据调用依赖库[darkskylib-master], [详细内容.](https://github.com/lukaskubis/darkskylib)  
>气溶胶数据: HDF文件, 读取依赖库[python_hdf](https://www.lfd.uci.edu/~gohlke/pythonlibs/)  
>插值: 插值库[fancyimpute](https://pypi.org/project/fancyimpute/)安装遭遇失败时, 如'failed build', 'get microsoft build tools from..', 可以先安装[scs, ecos](https://www.lfd.uci.edu/~gohlke/pythonlibs/)  
#### [程序](https://github.com/xunchanglu0901/MODIS04_3K_PM25/tree/master/%E6%89%A9%E5%B1%95%E5%BA%93%E5%92%8C%E6%89%A9%E5%B1%95%E7%A8%8B%E5%BA%8F/%E7%A8%8B%E5%BA%8F)
>学习过程中遇到的问题, 对应的整理出了解决方法..
  
# 感谢
>非环境科学领域专业学生, 对该内容感兴趣便着手进行了.  
>代码上存在运算效率不够高, 繁琐的问题也欢迎大家给我提建议, 十分感谢.  
>加油, (ง •_•)ง~   
>>欢迎大家来交流哈~ xunchanglu@vip.163.com
