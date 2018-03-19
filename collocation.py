# coding: utf-8
__author__ = 'wangpeng'

'''
Description:  静止和极轨通用匹配程序
Author:       wangpeng
Date:         2017-08-10
version:      1.0.1_beat
Input:        yaml格式配置文件
Output:       hdf5格式匹配文件  (^_^)
'''

# 引用系统库
import sys, os, re, h5py, yaml, time, gc
import numpy as np
from datetime import datetime
from configobj import ConfigObj

# 引用自编库
from PB.DRC.pb_drc_mersi2_L1_new import CLASS_MERSI2_L1_1000M
from PB.DRC.pb_drc_viirs_L1_new import CLASS_VIIRS_L1
from PB.DRC.pb_drc_modis_L1_new import CLASS_MODIS_L1
from PB.DRC.pb_drc_iasi_L1 import CLASS_IASI_L1
from PB.DRC.pb_drc_cris_L1 import CLASS_CRIS_L1
from PB.DRC.pb_drc_hmw8_new import CLASS_HMW8_L1
from DP.dp_collcation_new2 import *
from DP.dp_prj_new import  prj_core
from DV import dv_map, dv_plt, dv_img

# 文件默认编码修改位utf-8
reload(sys)
sys.setdefaultencoding('utf-8')
# 配置文件信息，设置为全局
MainPath, MainFile = os.path.split(os.path.realpath(__file__))

class ReadYaml():

    def __init__(self, inFile):
        '''
        读取yaml格式配置文件
        '''
        if not os.path.isfile(inFile):
            print 'Not Found %s' % inFile
            sys.exit(-1)

        with open(inFile, 'r') as stream:
            cfg = yaml.load(stream)
        self.sat1 = cfg['INFO']['sat1']
        self.sensor1 = cfg['INFO']['sensor1']
        self.sat2 = cfg['INFO']['sat2']
        self.sensor2 = cfg['INFO']['sensor2']
        self.ymd = cfg['INFO']['ymd']
#         self.hms = cfg['INFO']['hms']

        self.ifile1 = cfg['PATH']['ipath1']
        self.ifile2 = cfg['PATH']['ipath2']
        self.ofile = cfg['PATH']['opath']

        self.cmd = cfg['PROJ']['cmd']
        self.col = cfg['PROJ']['col']
        self.row = cfg['PROJ']['row']
        self.res = cfg['PROJ']['res']

def write_dclc(DCLC, ICFG, MCFG):

    print u'输出产品'
    for band in MCFG.chan1:
        idx = np.where(DCLC.MaskFine[band] > 0)
        DCLC_nums = len(idx[0])
        if (DCLC_nums > 0):
            break
    if DCLC_nums == 0:
        print('colloc point is zero')
        sys.exit(-1)

    # 根据卫星性质来命名数据集，固定标识，避免命名烦恼 烦恼 烦恼
    NameHead1 = 'S1_'
    NameHead2 = 'S2_'
    # 创建文件夹
    MainPath, MainFile = os.path.split(ICFG.ofile)
    if not os.path.isdir(MainPath):
        os.makedirs(MainPath)

    # 创建hdf5文件
    h5File_W = h5py.File(ICFG.ofile, 'w')

    if DCLC.spec_MaskRough_value is not None:
        dset = h5File_W.create_dataset('%sSpec_MaskRough_value' % (NameHead2), dtype='f4', data=DCLC.spec_MaskRough_value, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('%sSpec_MaskRough_row' % (NameHead2), dtype='i2', data=DCLC.spec_MaskRough_row, compression='gzip', compression_opts=5, shuffle=True)
        h5File_W.create_dataset('/%sSpec_MaskRough_col' % (NameHead2), dtype='i2', data=DCLC.spec_MaskRough_col, compression='gzip', compression_opts=5, shuffle=True)

        dset.attrs.create('Long_name', 'Record spectral lines obtained from MaskRough dataset', shape=(1,), dtype='S64')
    # 生成 h5,首先写入全局变量
    # 第一颗传感器的全局数据信息
    h5File_W.create_dataset('%sLon' % NameHead1, dtype='f4', data=DCLC.Lon1, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sLat' % NameHead1, dtype='f4', data=DCLC.Lat1, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sTime' % NameHead1, dtype='f4', data=DCLC.Time1, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSatA' % NameHead1, dtype='f4', data=DCLC.SatA1, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSatZ' % NameHead1, dtype='f4', data=DCLC.SatZ1, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSoA' % NameHead1, dtype='f4', data=DCLC.SunA1, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSoZ' % NameHead1, dtype='f4', data=DCLC.SunZ1, compression='gzip', compression_opts=5, shuffle=True)

    if DCLC.LandCover1 is not None:
        h5File_W.create_dataset('%sLandCover' % NameHead1, dtype='f4', data=DCLC.LandCover1, compression='gzip', compression_opts=5, shuffle=True)
    if DCLC.LandSeaMask1 is not None:
        h5File_W.create_dataset('%sLandSeaMask' % NameHead1, dtype='f4', data=DCLC.LandSeaMask1, compression='gzip', compression_opts=5, shuffle=True)

    # 第二颗传感器的全局数据信息
    h5File_W.create_dataset('%sLon' % NameHead2, dtype='f4', data=DCLC.Lon2, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sLat' % NameHead2, dtype='f4', data=DCLC.Lat2, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sTime' % NameHead2, dtype='f4', data=DCLC.Time2, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSatA' % NameHead2, dtype='f4', data=DCLC.SatA2, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSatZ' % NameHead2, dtype='f4', data=DCLC.SatZ2, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSoA' % NameHead2, dtype='f4', data=DCLC.SunA2, compression='gzip', compression_opts=5, shuffle=True)
    h5File_W.create_dataset('%sSoZ' % NameHead2, dtype='f4', data=DCLC.SunZ2, compression='gzip', compression_opts=5, shuffle=True)

    if DCLC.LandCover2 is not None:
        h5File_W.create_dataset('%sLandCover' % NameHead2, dtype='f4', data=DCLC.LandCover2, compression='gzip', compression_opts=5, shuffle=True)
    if DCLC.LandSeaMask2 is not None:
        h5File_W.create_dataset('%sLandSeaMask' % NameHead2, dtype='f4', data=DCLC.LandSeaMask2, compression='gzip', compression_opts=5, shuffle=True)

    # 写入掩码属性
    dset = h5File_W.create_dataset('MaskRough', dtype='u2', data=DCLC.MaskRough, compression='gzip', compression_opts=5, shuffle=True)
    dset.attrs.create('Long_name', 'after time and angle collocation', shape=(1,), dtype='S32')

    # 写入1通道数据信息
    for Band in MCFG.chan1:
        ###################### 第一颗传感器通道数据 ########################

        if DCLC.SV1[Band] is not None:
            h5File_W.create_dataset('/%s/%sSV' % (Band, NameHead1), dtype='f4', data=DCLC.SV1[Band], compression='gzip', compression_opts=5, shuffle=True)
        if DCLC.BB1[Band] is not None:
            h5File_W.create_dataset('/%s/%sBB' % (Band, NameHead1), dtype='f4', data=DCLC.BB1[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovDnMean1[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovDnMean' % (Band, NameHead1), dtype='f4', data=DCLC.FovDnMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovDnStd' % (Band, NameHead1), dtype='f4', data=DCLC.FovDnStd1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvDnMean' % (Band, NameHead1), dtype='f4', data=DCLC.EnvDnMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvDnStd' % (Band, NameHead1), dtype='f4', data=DCLC.EnvDnStd1[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovRefMean1[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovRefMean' % (Band, NameHead1), dtype='f4', data=DCLC.FovRefMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovRefStd' % (Band, NameHead1), dtype='f4', data=DCLC.FovRefStd1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRefMean' % (Band, NameHead1), dtype='f4', data=DCLC.EnvRefMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRefStd' % (Band, NameHead1), dtype='f4', data=DCLC.EnvRefStd1[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovRadMean1[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovRadMean' % (Band, NameHead1), dtype='f4', data=DCLC.FovRadMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovRadStd' % (Band, NameHead1), dtype='f4', data=DCLC.FovRadStd1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRadMean' % (Band, NameHead1), dtype='f4', data=DCLC.EnvRadMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRadStd' % (Band, NameHead1), dtype='f4', data=DCLC.EnvRadStd1[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovTbbMean1[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovTbbMean' % (Band, NameHead1), dtype='f4', data=DCLC.FovTbbMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovTbbStd' % (Band, NameHead1), dtype='f4', data=DCLC.FovTbbStd1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvTbbMean' % (Band, NameHead1), dtype='f4', data=DCLC.EnvTbbMean1[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvTbbStd' % (Band, NameHead1), dtype='f4', data=DCLC.EnvTbbStd1[Band], compression='gzip', compression_opts=5, shuffle=True)

        ###################### 第二颗传感器通道数据 ########################
        if DCLC.SV2[Band] is not None:
            h5File_W.create_dataset('/%s/%sSV' % (Band, NameHead2), dtype='f4', data=DCLC.SV2[Band], compression='gzip', compression_opts=5, shuffle=True)
        if DCLC.BB2[Band] is not None:
            h5File_W.create_dataset('/%s/%sBB' % (Band, NameHead2), dtype='f4', data=DCLC.BB2[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovDnMean2[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovDnMean' % (Band, NameHead2), dtype='f4', data=DCLC.FovDnMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovDnStd' % (Band, NameHead2), dtype='f4', data=DCLC.FovDnStd2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvDnMean' % (Band, NameHead2), dtype='f4', data=DCLC.EnvDnMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvDnStd' % (Band, NameHead2), dtype='f4', data=DCLC.EnvDnStd2[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovRefMean2[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovRefMean' % (Band, NameHead2), dtype='f4', data=DCLC.FovRefMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovRefStd' % (Band, NameHead2), dtype='f4', data=DCLC.FovRefStd2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRefMean' % (Band, NameHead2), dtype='f4', data=DCLC.EnvRefMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRefStd' % (Band, NameHead2), dtype='f4', data=DCLC.EnvRefStd2[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovRadMean2[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovRadMean' % (Band, NameHead2), dtype='f4', data=DCLC.FovRadMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovRadStd' % (Band, NameHead2), dtype='f4', data=DCLC.FovRadStd2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRadMean' % (Band, NameHead2), dtype='f4', data=DCLC.EnvRadMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvRadStd' % (Band, NameHead2), dtype='f4', data=DCLC.EnvRadStd2[Band], compression='gzip', compression_opts=5, shuffle=True)

        if DCLC.FovTbbMean2[Band] is not None:
            h5File_W.create_dataset('/%s/%sFovTbbMean' % (Band, NameHead2), dtype='f4', data=DCLC.FovTbbMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sFovTbbStd' % (Band, NameHead2), dtype='f4', data=DCLC.FovTbbStd2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvTbbMean' % (Band, NameHead2), dtype='f4', data=DCLC.EnvTbbMean2[Band], compression='gzip', compression_opts=5, shuffle=True)
            h5File_W.create_dataset('/%s/%sEnvTbbStd' % (Band, NameHead2), dtype='f4', data=DCLC.EnvTbbStd2[Band], compression='gzip', compression_opts=5, shuffle=True)

        dset = h5File_W.create_dataset('/%s/MaskFine' % Band, dtype='u2', data=DCLC.MaskFine[Band], compression='gzip', compression_opts=5, shuffle=True)
        dset.attrs.create('Long_name', 'after scene homogenous collocation', shape=(1,), dtype='S32')
    h5File_W.close()


def regression(x, y, min, max, flag, ICFG, MCFG, Band):

    # FY4分布
    MainPath, MainFile = os.path.split(ICFG.ofile)
    if not os.path.isdir(MainPath):
        os.makedirs(MainPath)

    meanbais = (np.mean(x - y) / np.mean(y)) * 100.

    p = dv_plt.dv_scatter(figsize=(7, 5))
    p.easyplot(x, y, None, None, marker='o', markersize=5)


    p.xlim_min = p.ylim_min = min
    p.xlim_max = p.ylim_max = max

    p.title = u'%s' % (ICFG.ymd)
    p.xlabel = u'%s %s %s' % (ICFG.sat1, ICFG.sensor1, flag)
    p.ylabel = u'%s %s %s' % (ICFG.sat2, ICFG.sensor2, flag)
    # 计算AB
    ab = np.polyfit(x, y, 1)
    p.regression(ab[0], ab[1], 'b')

    # 计算相关性
    p.show_leg = True
    r = np.corrcoef(x, y)
    rr = r[0][1] * r[0][1]
    nums = len(x)
    # 绘制散点
    strlist = [[r'$%0.4fx%+0.4f (R=%0.4f) $' % (ab[0], ab[1], rr), r'count:%d' % nums, r'%sMeanBias: %0.4f' % (flag, meanbais)]]
    p.annotate(strlist, 'left', 'r')
    ofile = os.path.join(MainPath, '%s+%s_%s+%s_%s_%s_%s.png' % (ICFG.sat1, ICFG.sensor1, ICFG.sat2, ICFG.sensor2, ICFG.ymd, Band, flag))
    p.savefig(ofile, dpi=300)

def draw_dclc(DCLC, ICFG, MCFG):

    print u'产品绘图'

    i = 0
    for Band in MCFG.chan1:
        idx = np.where(DCLC.MaskFine[Band] > 0)
        if DCLC.FovRefMean1[Band] is not None:
            x = DCLC.FovRefMean1[Band][idx]
            y = DCLC.FovRefMean2[Band][idx]
            if len(x) < 2:
                continue
            flag = 'Ref'
            if MCFG.AutoRange:
                min = np.min([np.min(x), np.min(y)])
                max = np.max([np.max(x), np.max(y)])
            else:
                min = MCFG.axis_ref[Band][0]
                max = MCFG.axis_ref[Band][1]

            regression(x, y, min, max, flag, ICFG, MCFG, Band)

        if DCLC.FovRadMean1[Band] is not None:
            x = DCLC.FovRadMean1[Band][idx]
            y = DCLC.FovRadMean2[Band][idx]
            if len(x) < 2:
                continue
            flag = 'Rad'
            print 'rad', Band, np.min(x), np.max(x), np.min(y), np.max(y)
            if MCFG.AutoRange:
                min = np.min([np.min(x), np.min(y)])
                max = np.max([np.max(x), np.max(y)])
            else:
                min = MCFG.axis_rad[Band][0]
                max = MCFG.axis_rad[Band][1]

            regression(x, y, min, max, flag, ICFG, MCFG, Band)

        if DCLC.FovTbbMean1[Band] is not None:
            x = DCLC.FovTbbMean1[Band][idx]
            y = DCLC.FovTbbMean2[Band][idx]
            if len(x) < 2:
                continue
            flag = 'Tbb'
            print 'tbb', Band, np.min(x), np.max(x), np.min(y), np.max(y)
            if MCFG.AutoRange:
                min = np.min([np.min(x), np.min(y)])
                max = np.max([np.max(x), np.max(y)])
            else:
                min = MCFG.axis_tbb[Band][0]
                max = MCFG.axis_tbb[Band][1]
            print 'tbb', Band,min,max
            regression(x, y, min, max, flag, ICFG, MCFG, Band)


def main(inYamlFile):

    T1 = datetime.now()


    ##########01 ICFG = 输入配置文件类 ##########
    ICFG = ReadYaml(inYamlFile)
    ##########02 MCFG = 阈值配置文件类
    modeFile = os.path.join(MainPath, 'COLLOC_%s_%s.yaml' % (ICFG.sensor1, ICFG.sensor2))
    MCFG = ReadModeYaml(modeFile)
    DCLC = COLLOC_COMM(ICFG.row, ICFG.col, MCFG.chan1)  # DCLC = DATA DCLC 匹配结果类
    ##########03 解析 第一颗传感器的L1数据 ##########
    for inFile in ICFG.ifile1:
        if 'MERSI' == ICFG.sensor1:
            D1 = CLASS_MERSI_L1()
            LutFile = os.path.join(MainPath, 'FY3C-MERSI-LUT-TB-RB.txt')
            D1.LutFile = LutFile
            D1.Load(inFile)
        elif 'VIRR' == ICFG.sensor1:
            D1 = CLASS_VIRR_L1()
            LutFile = os.path.join(MainPath, 'FY3C-VIRR-LUT-TB-RB.txt')
            D1.LutFile = LutFile
            D1.Load(inFile)
        elif 'MERSI2' == ICFG.sensor1:
            D1 = CLASS_MERSI2_L1_1000M()
            LutFile = ''
            D1.LutFile = LutFile
            D1.Load(inFile)
        elif 'AHI' == ICFG.sensor1:
            D1 = CLASS_HMW8_L1()
            D1.Load(inFile)
            geoFile = 'fygatNAV.Himawari08.xxxxxxx.000001_minmin.hdf'
            D1.Loadgeo(geoFile)
        else:
            print 'sensor1:%s not support' % ICFG.sensor1


        ##########04 投影，简历查找表  ##########
        P1 = prj_core(ICFG.cmd, ICFG.res, row=ICFG.row, col=ICFG.col)
        P1.create_lut(D1.Lons, D1.Lats)

        ##########05 解析 第二颗传感器的L1数据   ##########
        for inFile2 in ICFG.ifile2:
            if 'MODIS' == ICFG.sensor2:
                D2 = CLASS_MODIS_L1()
                D2.Load(inFile2)
            elif 'VIIRS' == ICFG.sensor2:
                D2 = CLASS_VIIRS_L1()
                D2.Load(inFile2)
            elif 'MERSI2' == ICFG.sensor2:
                D2 = CLASS_MERSI2_L1_1000M()
                D2.Load(inFile2)
            elif 'IASI' == ICFG.sensor2:
                D2 = CLASS_IASI_L1(MCFG.chan1)
                D2.Load(inFile2)
                D2.get_rad_tbb(D1, MCFG.chan1)

            elif 'CRIS' == ICFG.sensor2:
                D2 = CLASS_CRIS_L1(MCFG.chan1)
                D2.Load(inFile2)
                D2.get_rad_tbb(D1, MCFG.chan1)

            ##########06 投影，简历查找表  ##########
            P2 = prj_core(ICFG.cmd, ICFG.res, row=ICFG.row, col=ICFG.col)
            P2.create_lut(D2.Lons, D2.Lats)
            ##########07 粗匹配 ##########
            DCLC.save_rough_data(P1, P2, D1, D2, MCFG)
    ##########08 精匹配 ##########
#     DCLC.write_mid_hdf5(ICFG, MCFG)

    DCLC.save_fine_data(MCFG)

    T2 = datetime.now()
    print 'colloc:', (T2 - T1).total_seconds()

    ##########09 输出匹配结果 ##########

    if  MCFG.rewrite:
        T1 = datetime.now()
        write_dclc(DCLC, ICFG, MCFG)
        T2 = datetime.now()
        print 'write:', (T2 - T1).total_seconds()
    ##########10 对结果进行绘图 ##########
    if  MCFG.drawmap:
        T1 = datetime.now()
        draw_dclc(DCLC, ICFG, MCFG)
        T2 = datetime.now()
        print 'map:', (T2 - T1).total_seconds()

if __name__ == '__main__':

    # 获取python输入参数，进行处理
    args = sys.argv[1:]
    if len(args) == 1:  # 跟参数，则处理输入的时段数据
        inYamlFile = args[0]
    else:
        print 'input args error exit'
        sys.exit(-1)

    # 统计运行时间
    T1 = datetime.now()
    main(inYamlFile)
    T2 = datetime.now()
    print 'times:', (T2 - T1).total_seconds()
