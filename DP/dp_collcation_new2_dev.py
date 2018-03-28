# coding: utf-8
'''
匹配类
@author: wangpeng
'''
import os, sys, yaml, h5py
import numpy as np
from dp_2d import rolling_2d_window_pro
from PB.pb_space import sun_glint_cal

class ReadModeYaml():
    '''
        读取yaml格式配置文件,解析匹配的传感器对的默认配置参数
    '''

    def __init__(self, inFile):

        if not os.path.isfile(inFile):
            print 'Not Found %s' % inFile
            sys.exit(-1)

        with open(inFile, 'r') as stream:
            cfg = yaml.load(stream)
        self.sensor1 = cfg['sensor1']
        self.sensor2 = cfg['sensor2']
        self.chan1 = cfg['chan1']
        self.chan2 = cfg['chan2']
        self.rewrite = cfg['rewrite']
        self.drawmap = cfg['drawmap']

        self.FovWind1 = tuple(cfg['FovWind1'])
        self.EnvWind1 = tuple(cfg['EnvWind1'])
        self.FovWind2 = tuple(cfg['FovWind2'])
        self.EnvWind2 = tuple(cfg['EnvWind2'])

        self.solglint_min = cfg['solglint_min']
        self.solzenith_max = cfg['solzenith_max']
        self.timedif_max = cfg['timedif_max']
        self.angledif_max = cfg['angledif_max']
        self.distdif_max = cfg['distdif_max']
        self.AutoRange = cfg['AutoRange']
        self.clear_band_ir = cfg['clear_band_ir']
        self.clear_min_ir = cfg['clear_min_ir']
        self.clear_band_vis = cfg['clear_band_vis']
        self.clear_max_vis = cfg['clear_max_vis']

        if 'write_spec' in cfg.keys():
            self.write_spec = cfg['write_spec']
        else:
            self.write_spec = None

        if 'axis_ref' in cfg.keys():
            self.axis_ref = cfg['axis_ref']
        if 'axis_rad' in cfg.keys():
            self.axis_rad = cfg['axis_rad']
        if 'axis_tbb' in cfg.keys():
            self.axis_tbb = cfg['axis_tbb']

        # 将通道阈值放入字典
        self.CH_threshold = {}
        for ch in self.chan1:
            if ch not in self.CH_threshold.keys():
                self.CH_threshold[ch] = {}
            for threshold in cfg[ch]:
                self.CH_threshold[ch][threshold] = cfg[ch][threshold]

class COLLOC_COMM():
    '''
    交叉匹配的公共类，首先初始化所有参数信息
    '''
    def __init__(self, row, col, BandLst):

        # 默认填充值 和 数据类型
        self.row = row
        self.col = col
        self.FillValue = -999.
        self.dtype = 'f4'

        # 投影后的全局变量信息
        self.Time1 = np.full((row, col), self.FillValue, self.dtype)
        self.Lon1 = np.full((row, col), self.FillValue, self.dtype)
        self.Lat1 = np.full((row, col), self.FillValue, self.dtype)
        self.SatA1 = np.full((row, col), self.FillValue, self.dtype)
        self.SatZ1 = np.full((row, col), self.FillValue, self.dtype)
        self.SunA1 = np.full((row, col), self.FillValue, self.dtype)
        self.SunZ1 = np.full((row, col), self.FillValue, self.dtype)

        self.LandCover1 = np.full((row, col), -999, 'i2')
        self.LandSeaMask1 = np.full((row, col), -999, 'i2')

        self.Time2 = np.full((row, col), self.FillValue, self.dtype)
        self.Lon2 = np.full((row, col), self.FillValue, self.dtype)
        self.Lat2 = np.full((row, col), self.FillValue, self.dtype)
        self.SatA2 = np.full((row, col), self.FillValue, self.dtype)
        self.SatZ2 = np.full((row, col), self.FillValue, self.dtype)
        self.SunA2 = np.full((row, col), self.FillValue, self.dtype)
        self.SunZ2 = np.full((row, col), self.FillValue, self.dtype)

        self.LandCover2 = np.full((row, col), -999, 'i2')
        self.LandSeaMask2 = np.full((row, col), -999, 'i2')

        # 高光谱信息
        self.spec_MaskRough_row = []  # 记录投影2维网格的行
        self.spec_MaskRough_col = []  # 记录投影2维网格的列
        self.spec_MaskRough_all = {}  # 记录所有光谱
        self.spec_MaskRough_value = None  # 记录粗掩码表的光谱

        self.MaskRough = np.full((row, col), 0, 'i1')
        self.PubIdx = np.full((row, col), 0, 'i1')

        # 定义各个物理属性字典
        self.MaskFine = {}

        # SAT1 FOV ENV
        self.FovDnMean1 = {}
        self.FovDnStd1 = {}
        self.FovRefMean1 = {}
        self.FovRefStd1 = {}
        self.FovRadMean1 = {}
        self.FovRadStd1 = {}
        self.FovTbbMean1 = {}
        self.FovTbbStd1 = {}

        self.EnvDnMean1 = {}
        self.EnvDnStd1 = {}
        self.EnvRefMean1 = {}
        self.EnvRefStd1 = {}
        self.EnvRadMean1 = {}
        self.EnvRadStd1 = {}
        self.EnvTbbMean1 = {}
        self.EnvTbbStd1 = {}
        self.SV1 = {}
        self.BB1 = {}

        # SAT2 FOV ENV
        self.FovDnMean2 = {}
        self.FovDnStd2 = {}
        self.FovRefMean2 = {}
        self.FovRefStd2 = {}
        self.FovRadMean2 = {}
        self.FovRadStd2 = {}
        self.FovTbbMean2 = {}
        self.FovTbbStd2 = {}

        self.EnvDnMean2 = {}
        self.EnvDnStd2 = {}
        self.EnvRefMean2 = {}
        self.EnvRefStd2 = {}
        self.EnvRadMean2 = {}
        self.EnvRadStd2 = {}
        self.EnvTbbMean2 = {}
        self.EnvTbbStd2 = {}
        self.SV2 = {}
        self.BB2 = {}

        # 初始化字典内的存放每个通道的数据空间
        for band in BandLst:
            self.MaskFine[band] = np.full((row, col), 0, 'i1')
            self.FovDnMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovDnStd1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRefMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRefStd1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRadMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRadStd1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovTbbMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovTbbStd1[band] = np.full((row, col), self.FillValue, self.dtype)

            self.EnvDnMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvDnStd1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRefMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRefStd1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRadMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRadStd1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvTbbMean1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvTbbStd1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.SV1[band] = np.full((row, col), self.FillValue, self.dtype)
            self.BB1[band] = np.full((row, col), self.FillValue, self.dtype)

            # SAT2 FOV ENV
            self.FovDnMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovDnStd2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRefMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRefStd2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRadMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovRadStd2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovTbbMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.FovTbbStd2[band] = np.full((row, col), self.FillValue, self.dtype)

            self.EnvDnMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvDnStd2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRefMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRefStd2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRadMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvRadStd2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvTbbMean2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.EnvTbbStd2[band] = np.full((row, col), self.FillValue, self.dtype)

            self.SV2[band] = np.full((row, col), self.FillValue, self.dtype)
            self.BB2[band] = np.full((row, col), self.FillValue, self.dtype)

    def reload_data(self, ICFG, MCFG):
        """
        :param modeCfg: 配置文件
        :return:
        """
        i_file = ICFG.ofile
        with h5py.File(i_file, 'r') as hdf5File:
            global_keys = hdf5File.keys()
            if 'MaskRough' in global_keys:
                self.MaskRough = hdf5File.get('MaskRough')[:]

            if 'S1_Time' in global_keys:
                self.Time1 = hdf5File.get('S1_Time')[:]
            if 'S1_Lon' in global_keys:
                self.Lon1 = hdf5File.get('S1_Lon')[:]
            if 'S1_Lat' in global_keys:
                self.Lat1 = hdf5File.get('S1_Lat')[:]
            if 'S1_SatA' in global_keys:
                self.SatA1 = hdf5File.get('S1_SatA')[:]
            if 'S1_SatZ' in global_keys:
                self.SatZ1 = hdf5File.get('S1_SatZ')[:]
            if 'S1_SunA' in global_keys:
                self.SunA1 = hdf5File.get('S1_SunA')[:]
            if 'S1_SunZ' in global_keys:
                self.SunZ1 = hdf5File.get('S1_SunZ')[:]
            if 'S1_LandCover' in global_keys:
                self.LandCover1 = hdf5File.get('S1_LandCover')[:]
            if 'S1_LandSeaMask' in global_keys:
                self.LandSeaMask1 = hdf5File.get('S1_LandSeaMask')[:]

            if 'S2_Time' in global_keys:
                self.Time2 = hdf5File.get('S2_Time')[:]
            if 'S2_Lon' in global_keys:
                self.Lon2 = hdf5File.get('S2_Lon')[:]
            if 'S2_Lat' in global_keys:
                self.Lat2 = hdf5File.get('S2_Lat')[:]
            if 'S2_SatA' in global_keys:
                self.SatA2 = hdf5File.get('S2_SatA')[:]
            if 'S2_SatZ' in global_keys:
                self.SatZ2 = hdf5File.get('S2_SatZ')[:]
            if 'S2_SunA' in global_keys:
                self.SunA2 = hdf5File.get('S2_SunA')[:]
            if 'S2_SunZ' in global_keys:
                self.SunZ2 = hdf5File.get('S2_SunZ')[:]
            if 'S2_LandCover' in global_keys:
                self.LandCover2 = hdf5File.get('S2_LandCover')[:]
            if 'S2_LandSeaMask' in global_keys:
                self.LandSeaMask2 = hdf5File.get('S2_LandSeaMask')[:]

            for band in MCFG['chan1']:
                band_keys = hdf5File.get(band).keys()
                if 'MaskFine' in band_keys:
                    self.MaskFine[band] = hdf5File.get(band)['MaskFine'][:]

                if 'S1_FovDnMean' in band_keys:
                    self.FovDnMean1[band] = hdf5File.get(band)['S1_FovDnMean'][:]
                if 'S1_FovRefMean' in band_keys:
                    self.FovRefMean1[band] = hdf5File.get(band)['S1_FovRefMean'][:]
                if 'S1_FovRefStd' in band_keys:
                    self.FovRefStd1[band] = hdf5File.get(band)['S1_FovRefStd'][:]
                if 'S1_FovRadMean' in band_keys:
                    self.FovRadMean1[band] = hdf5File.get(band)['S1_FovRadMean'][:]
                if 'S1_FovRadStd' in band_keys:
                    self.FovRadStd1[band] = hdf5File.get(band)['S1_FovRadStd'][:]
                if 'S1_FovTbbMean' in band_keys:
                    self.FovTbbMean1[band] = hdf5File.get(band)['S1_FovTbbMean'][:]
                if 'S1_FovTbbStd' in band_keys:
                    self.FovTbbStd1[band] = hdf5File.get(band)['S1_FovTbbStd'][:]

                if 'S1_EnvDnMean' in band_keys:
                    self.EnvDnMean1[band] = hdf5File.get(band)['S1_EnvDnMean'][:]
                if 'S1_EnvDnStd' in band_keys:
                    self.EnvDnStd1[band] = hdf5File.get(band)['S1_EnvDnStd'][:]
                if 'S1_EnvRefMean' in band_keys:
                    self.EnvRefMean1[band] = hdf5File.get(band)['S1_EnvRefMean'][:]
                if 'S1_EnvRefStd' in band_keys:
                    self.EnvRefStd1[band] = hdf5File.get(band)['S1_EnvRefStd'][:]
                if 'S1_EnvRadMean' in band_keys:
                    self.EnvRadMean1[band] = hdf5File.get(band)['S1_EnvRadMean'][:]
                if 'S1_EnvRadStd' in band_keys:
                    self.EnvRadStd1[band] = hdf5File.get(band)['S1_EnvRadStd'][:]
                if 'S1_EnvTbbMean' in band_keys:
                    self.EnvTbbMean1[band] = hdf5File.get(band)['S1_EnvTbbMean'][:]
                if 'S1_EnvTbbStd' in band_keys:
                    self.EnvTbbStd1[band] = hdf5File.get(band)['S1_EnvTbbStd'][:]
                if 'S1_SV' in band_keys:
                    self.SV1[band] = hdf5File.get(band)['S1_SV'][:]
                if 'S1_BB' in band_keys:
                    self.BB1[band] = hdf5File.get(band)['S1_BB'][:]

                if 'S2_FovDnMean' in band_keys:
                    self.FovDnMean2[band] = hdf5File.get(band)['S2_FovDnMean'][:]
                if 'S2_FovRefMean' in band_keys:
                    self.FovRefMean2[band] = hdf5File.get(band)['S2_FovRefMean'][:]
                if 'S2_FovRefStd' in band_keys:
                    self.FovRefStd2[band] = hdf5File.get(band)['S2_FovRefStd'][:]
                if 'S2_FovRadMean' in band_keys:
                    self.FovRadMean2[band] = hdf5File.get(band)['S2_FovRadMean'][:]
                if 'S2_FovRadStd' in band_keys:
                    self.FovRadStd2[band] = hdf5File.get(band)['S2_FovRadStd'][:]
                if 'S2_FovTbbMean' in band_keys:
                    self.FovTbbMean2[band] = hdf5File.get(band)['S2_FovTbbMean'][:]
                if 'S2_FovTbbStd' in band_keys:
                    self.FovTbbStd2[band] = hdf5File.get(band)['S2_FovTbbStd'][:]

                if 'S2_EnvDnMean' in band_keys:
                    self.EnvDnMean2[band] = hdf5File.get(band)['S2_EnvDnMean'][:]
                if 'S2_EnvDnStd' in band_keys:
                    self.EnvDnStd2[band] = hdf5File.get(band)['S2_EnvDnStd'][:]
                if 'S2_EnvRefMean' in band_keys:
                    self.EnvRefMean2[band] = hdf5File.get(band)['S2_EnvRefMean'][:]
                if 'S2_EnvRefStd' in band_keys:
                    self.EnvRefStd2[band] = hdf5File.get(band)['S2_EnvRefStd'][:]
                if 'S2_EnvRadMean' in band_keys:
                    self.EnvRadMean2[band] = hdf5File.get(band)['S2_EnvRadMean'][:]
                if 'S2_EnvRadStd' in band_keys:
                    self.EnvRadStd2[band] = hdf5File.get(band)['S2_EnvRadStd'][:]
                if 'S2_EnvTbbMean' in band_keys:
                    self.EnvTbbMean2[band] = hdf5File.get(band)['S2_EnvTbbMean'][:]
                if 'S2_EnvTbbStd' in band_keys:
                    self.EnvTbbStd2[band] = hdf5File.get(band)['S2_EnvTbbStd'][:]
                if 'S2_SV' in band_keys:
                    self.SV2[band] = hdf5File.get(band)['S2_SV'][:]
                if 'S2_BB' in band_keys:
                    self.BB2[band] = hdf5File.get(band)['S2_BB'][:]

    def save_rough_data(self, P1, P2, D1, D2, modeCfg):
        '''
        第一轮匹配，根据查找表进行数据的mean和std计算，并且对全局物理量复制（角度，经纬度，时间等）
        '''
        print u'对公共区域位置进行数据赋值......'
        # 公共的投影区域位置信息
        condition = np.logical_and(P1.lut_i > 0 , P2.lut_i > 0)
        idx = np.where(condition)
        # 记录粗匹配点
        self.PubIdx[idx] = 1
        print u'FY LEO 公共区域匹配点个数 %d' % len(idx[0])
        # 粗匹配点没有则返回
        if  len(idx[0]) == 0:
            return

        # 投影后网格，公共区域的投影后数据的行列
        p_i = idx[0]
        p_j = idx[1]


        # 投影后网格，公共区域的投影后 传感器1 和 传感器2 数据的行列
        i1 = P1.lut_i[idx]
        j1 = P1.lut_j[idx]
        i2 = P2.lut_i[idx]
        j2 = P2.lut_j[idx]

        # 保存传感器1 的投影公共数据信息
        self.Time1[idx] = D1.Time[i1, j1]
        self.Lon1[idx] = D1.Lons[i1, j1]
        self.Lat1[idx] = D1.Lats[i1, j1]
        self.SatA1[idx] = D1.satAzimuth[i1, j1]
        self.SatZ1[idx] = D1.satZenith[i1, j1]
        self.SunA1[idx] = D1.sunAzimuth[i1, j1]
        self.SunZ1[idx] = D1.sunZenith[i1, j1]

        if D1.LandCover is not None:
            self.LandCover1[idx] = D1.LandCover[i1, j1]
        else:
            self.LandCover1 = None
        if D1.LandSeaMask is not None:
            self.LandSeaMask1[idx] = D1.LandSeaMask[i1, j1]
        else:
            self.LandSeaMask1 = None

        # 保存传感器2 的投影公共数据信息
        self.Time2[idx] = D2.Time[i2, j2]
        self.Lon2[idx] = D2.Lons[i2, j2]
        self.Lat2[idx] = D2.Lats[i2, j2]
        self.SatA2[idx] = D2.satAzimuth[i2, j2]
        self.SatZ2[idx] = D2.satZenith[i2, j2]
        self.SunA2[idx] = D2.sunAzimuth[i2, j2]
        self.SunZ2[idx] = D2.sunZenith[i2, j2]

        if D2.LandCover is not None:
            self.LandCover2[idx] = D2.LandCover[i2, j2]
        else:
            self.LandCover2 = None
        if D2.LandSeaMask is not None:
            self.LandSeaMask2[idx] = D2.LandSeaMask[i2, j2]
        else:
            self.LandSeaMask2 = None

        if modeCfg.write_spec:
            # 2维下标转1维下标
            idx_1d = np.ravel_multi_index(idx, (self.row, self.col))
            for i in xrange(len(idx_1d)):
                self.spec_MaskRough_all[idx_1d[i]] = D2.radiance[i2[i], :]

        ############# sat1 各项值计算 #############
        for Band1 in modeCfg.chan1:
            index = modeCfg.chan1.index(Band1)
            Band2 = modeCfg.chan2[index]
            ############# sat1 DN #############
            if Band1 in D1.DN.keys():
                # sat1 Fov和Env dn的mean和std
                data = D1.DN['%s' % Band1]
                # 计算各个通道的投影后数据位置对应原始数据位置点的指定范围的均值和std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind1, i1, j1, p_i, p_j)
                self.FovDnMean1[Band1][pi, pj] = mean
                self.FovDnStd1[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind1, i1, j1, p_i, p_j)
                self.EnvDnMean1[Band1][pi, pj] = mean
                self.EnvDnStd1[Band1][pi, pj] = std
            else:
                self.FovDnMean1[Band1] = None
                self.FovDnStd1[Band1] = None
                self.EnvDnMean1[Band1] = None
                self.EnvDnStd1[Band1] = None

            ############# sat1 Ref #############
            if Band1 in D1.Ref.keys():
                # sat1 Fov和Env Ref的mean和std
                data = D1.Ref['%s' % Band1]
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind1, i1, j1, p_i, p_j)
                self.FovRefMean1[Band1][pi, pj] = mean
                self.FovRefStd1[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind1, i1, j1, p_i, p_j)
                self.EnvRefMean1[Band1][pi, pj] = mean
                self.EnvRefStd1[Band1][pi, pj] = std
            else:
                self.FovRefMean1[Band1] = None
                self.FovRefStd1[Band1] = None
                self.EnvRefMean1[Band1] = None
                self.EnvRefStd1[Band1] = None

            ############# sat1 Rad #############
            if Band1 in D1.Rad.keys():
                # sat1 Fov和Env Ref的mean和std
                data = D1.Rad['%s' % Band1]
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind1, i1, j1, p_i, p_j)
                self.FovRadMean1[Band1][pi, pj] = mean
                self.FovRadStd1[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind1, i1, j1, p_i, p_j)
                self.EnvRadMean1[Band1][pi, pj] = mean
                self.EnvRadStd1[Band1][pi, pj] = std
            else:
                self.FovRadMean1[Band1] = None
                self.FovRadStd1[Band1] = None
                self.EnvRadMean1[Band1] = None
                self.EnvRadStd1[Band1] = None

            ############# sat1 Tbb #############
            if Band1 in D1.Tbb.keys():
                # sat1 Fov和Env Ref的mean和std
                data = D1.Tbb['%s' % Band1]
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind1, i1, j1, p_i, p_j)
                self.FovTbbMean1[Band1][pi, pj] = mean
                self.FovTbbStd1[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind1, i1, j1, p_i, p_j)
                self.EnvTbbMean1[Band1][pi, pj] = mean
                self.EnvTbbStd1[Band1][pi, pj] = std
            else:
                self.FovTbbMean1[Band1] = None
                self.FovTbbStd1[Band1] = None
                self.EnvTbbMean1[Band1] = None
                self.EnvTbbStd1[Band1] = None

            # sat1 sv和 bb的赋值
            if D1.SV[Band1] is not None:
                self.SV1[Band1][p_i, p_j] = D1.SV[Band1][i1, j1]
            else:
                self.SV1[Band1] = None

            if D1.BB[Band1] is not None:
                self.BB1[Band1][p_i, p_j] = D1.BB[Band1][i1, j1]
            else:
                self.BB1[Band1] = None

            ############# sat2 各项值计算 #############
            ############# sat2 DN #############
            if Band2 in D2.DN.keys():
                # sat1 Fov和Env dn的mean和std
                data = D2.DN['%s' % Band2]
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind2, i2, j2, p_i, p_j)
                self.FovDnMean2[Band1][pi, pj] = mean
                self.FovDnStd2[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind2, i2, j2, p_i, p_j)
                self.EnvDnMean2[Band1][pi, pj] = mean
                self.EnvDnStd2[Band1][pi, pj] = std
            else:
                print 'sat2 not dn'
                self.FovDnMean2[Band1] = None
                self.FovDnStd2[Band1] = None
                self.EnvDnMean2[Band1] = None
                self.EnvDnStd2[Band1] = None

            ############# sat2 Ref #############
            if Band2 in D2.Ref.keys():
                # sat1 Fov和Env Ref的mean和std
                data = D2.Ref['%s' % Band2]
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind2, i2, j2, p_i, p_j)
                self.FovRefMean2[Band1][pi, pj] = mean
                self.FovRefStd2[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind2, i2, j2, p_i, p_j)
                self.EnvRefMean2[Band1][pi, pj] = mean
                self.EnvRefStd2[Band1][pi, pj] = std
            else:
                self.FovRefMean2[Band1] = None
                self.FovRefStd2[Band1] = None
                self.EnvRefMean2[Band1] = None
                self.EnvRefStd2[Band1] = None

            ############# sat1 Rad #############
            if Band2 in D2.Rad.keys():
                # sat1 Fov和Env Ref的mean和std
                data = D2.Rad['%s' % Band2]
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind2, i2, j2, p_i, p_j)
                self.FovRadMean2[Band1][pi, pj] = mean
                self.FovRadStd2[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind2, i2, j2, p_i, p_j)
                self.EnvRadMean2[Band1][pi, pj] = mean
                self.EnvRadStd2[Band1][pi, pj] = std
            else:
                self.FovRadMean2[Band1] = None
                self.FovRadStd2[Band1] = None
                self.EnvRadMean2[Band1] = None
                self.EnvRadStd2[Band1] = None

            ############# sat2 Tbb #############
            if Band2 in D2.Tbb.keys():
                # sat1 Fov和Env Ref的mean和std
                data = D2.Tbb['%s' % Band2]
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.FovWind2, i2, j2, p_i, p_j)
                self.FovTbbMean2[Band1][pi, pj] = mean
                self.FovTbbStd2[Band1][pi, pj] = std
                mean, std , pi, pj = rolling_2d_window_pro(data, modeCfg.EnvWind2, i2, j2, p_i, p_j)
                self.EnvTbbMean2[Band1][pi, pj] = mean
                self.EnvTbbStd2[Band1][pi, pj] = std
            else:
                self.FovTbbMean2[Band1] = None
                self.FovTbbStd2[Band1] = None
                self.EnvTbbMean2[Band1] = None
                self.EnvTbbStd2[Band1] = None


            # sat1 sv和 bb的赋值
            if D2.SV[Band2] is not None:
                self.SV2[Band1][p_i, p_j] = D2.SV[Band2][i2, j2]
            else:
                self.SV2[Band1] = None

            if D2.BB[Band2] is not None:
                self.BB2[Band1][p_i, p_j] = D2.BB[Band2][i2, j2]
            else:
                self.BB2[Band1] = None

    def save_fine_data(self, modeCfg):
        '''
        第二轮匹配，根据各通道的的mean和std计以为，角度和距离等进行精细化过滤
        '''

        # 最终的公共匹配点数量
        idx = np.where(self.PubIdx > 0)
        if  len(idx[0]) == 0:
            return
        print u'所有粗匹配点数目 ', len(idx[0])

        ############### 计算共同区域的距离差 #########
        disDiff = np.full_like(self.Time1, '-1', dtype='i2')
        a = np.power(self.Lon2[idx] - self.Lon1[idx], 2)
        b = np.power(self.Lat2[idx] - self.Lat1[idx], 2)
        disDiff[idx] = np.sqrt(a + b) * 100.

        idx_Rough = np.logical_and(disDiff < modeCfg.distdif_max, disDiff >= 0)
        idx1 = np.where(idx_Rough)
        print u'1. 距离过滤后剩余点 ', len(idx1[0])


        timeDiff = np.abs(self.Time1 - self.Time2)

        idx_Rough = np.logical_and(idx_Rough, timeDiff <= modeCfg.timedif_max)
        idx1 = np.where(idx_Rough)
        print u'2. 时间过滤后剩余点 ', len(idx1[0])
        ############### 过滤太阳天顶角 ###############
        idx_Rough = np.logical_and(idx_Rough , self.SunZ1 <= modeCfg.solzenith_max)
        idx_Rough = np.logical_and(idx_Rough , self.SunZ2 <= modeCfg.solzenith_max)
        idx1 = np.where(idx_Rough)
        print u'3. 太阳天顶角过滤后剩余点 ', len(idx1[0])

        ############### 计算耀斑角 ###############
        glint1 = np.full_like(self.SatZ1, -999.)
        glint2 = np.full_like(self.SatZ1, -999.)
        print 'self.SatA1[idx]=' , np.nanmin(self.SatA1[idx]), np.nanmax(self.SatA1[idx])
        print 'self.SatZ1[idx]=' , np.nanmin(self.SatZ1[idx]), np.nanmax(self.SatZ1[idx])
        print 'self.SunA1[idx]=' , np.min(self.SunA1[idx]), np.max(self.SunA1[idx])
        print 'self.SunZ1[idx]=' , np.min(self.SunZ1[idx]), np.max(self.SunZ1[idx])

        glint1[idx] = sun_glint_cal(self.SatA1[idx], self.SatZ1[idx], self.SunA1[idx], self.SunZ1[idx])
        glint2[idx] = sun_glint_cal(self.SatA2[idx], self.SatZ2[idx], self.SunA2[idx], self.SunZ2[idx])
        idx_Rough = np.logical_and(idx_Rough, glint1 > modeCfg.solglint_min)
        idx_Rough = np.logical_and(idx_Rough, glint2 > modeCfg.solglint_min)

        idx1 = np.where(idx_Rough)
        print u'4. 太阳耀斑角过滤后剩余点 ', len(idx1[0])

        ############### 角度均匀性 #################
        SatZRaio = np.full_like(self.Time1, 9999)
        SatZ1 = np.cos(self.SatZ1[idx] * np.pi / 180.)
        SatZ2 = np.cos(self.SatZ2[idx] * np.pi / 180.)
        SatZRaio[idx] = np.abs(SatZ1 / SatZ2 - 1.)

        idx_Rough = np.logical_and(idx_Rough, SatZRaio <= modeCfg.angledif_max)
        idx1 = np.where(idx_Rough)
        print u'5. 卫星天顶角均匀性过滤后剩余点 ', len(idx1[0])
        self.MaskRough[idx1] = 1

        # 添加spec, 粗匹配后剩余的点是要记录光谱谱线的。。。2维转1维下标
        idx_1d = np.ravel_multi_index(idx1, (self.row, self.col))

        if modeCfg.write_spec:
            # 定义spec_MaskRough_value 然后记录需要保存的谱线
            self.spec_MaskRough_value = []
            for i in idx_1d:
                self.spec_MaskRough_value.append(self.spec_MaskRough_all[i])
            self.spec_MaskRough_value = np.array(self.spec_MaskRough_value)

            # 记录根据MaskRough表记录的格点信息
            self.spec_MaskRough_row = idx1[0]
            self.spec_MaskRough_col = idx1[1]

        for Band1 in modeCfg.chan1:
            th_vaue_max = modeCfg.CH_threshold[Band1]['value_max']
            th1 = modeCfg.CH_threshold[Band1]['angledif_max']
            th2 = modeCfg.CH_threshold[Band1]['homodif_fov_max']
            th3 = modeCfg.CH_threshold[Band1]['homodif_env_max']
            th4 = modeCfg.CH_threshold[Band1]['homodif_fov_env_max']

            th_cld1 = modeCfg.CH_threshold[Band1]['cld_angledif_max']
            th_cld2 = modeCfg.CH_threshold[Band1]['cld_homodif_fov_max']
            th_cld3 = modeCfg.CH_threshold[Band1]['cld_homodif_env_max']
            th_cld4 = modeCfg.CH_threshold[Band1]['cld_homodif_fov_env_max']

            flag = 0
            # 如果 rad和tbb都有就用rad 做均匀性判断
            if (self.FovRadMean1[Band1] is not None) and (self.FovTbbMean1[Band1]is not None):
                flag = 'ir'
                # 固定通道值用于检测红外晴空和云
                irValue = self.FovTbbMean1[modeCfg.clear_band_ir]
                homoFov1 = np.abs(self.FovRadStd1[Band1] / self.FovRadMean1[Band1])
                homoEnv1 = np.abs(self.EnvRadStd1[Band1] / self.EnvRadMean1[Band1])
                homoFovEnv1 = np.abs(self.FovRadMean1[Band1] / self.EnvRadMean1[Band1] - 1)
                homoValue1 = self.FovRadMean1[Band1]
                homoFov2 = np.abs(self.FovRadStd2[Band1] / self.FovRadMean2[Band1])
                homoEnv2 = np.abs(self.EnvRadStd2[Band1] / self.EnvRadMean2[Band1])
                homoFovEnv2 = np.abs(self.FovRadMean2[Band1] / self.EnvRadMean2[Band1] - 1)
                homoValue2 = self.FovRadMean2[Band1]
            # 如果只有 tbb 就用tbb
            elif (self.FovTbbMean1[Band1] is not None) and (self.FovRadMean1[Band1] is None):
                flag = 'ir'
                # 固定通道值用于检测红外晴空和云
                irValue = self.FovTbbMean1[modeCfg.clear_band_ir]
                homoFov1 = np.abs(self.FovTbbStd1[Band1] / self.FovTbbMean1[Band1])
                homoEnv1 = np.abs(self.EnvTbbStd1[Band1] / self.EnvTbbMean1[Band1])
                homoFovEnv1 = np.abs(self.FovTbbMean1[Band1] / self.EnvTbbMean1[Band1] - 1)
                homoValue1 = self.FovTbbMean1[Band1]
                homoFov2 = np.abs(self.FovTbbStd2[Band1] / self.FovTbbMean2[Band1])
                homoEnv2 = np.abs(self.EnvTbbStd2[Band1] / self.EnvTbbMean2[Band1])
                homoFovEnv2 = np.abs(self.FovTbbMean2[Band1] / self.EnvTbbMean2[Band1] - 1)
                homoValue2 = self.FovTbbMean2[Band1]
            elif self.FovRefMean1[Band1] is not None:
                flag = 'vis'
                # 固定通道值用于检测可见晴空和云
                visValue = self.FovRefMean1[modeCfg.clear_band_vis]
                homoFov1 = np.abs(self.FovRefStd1[Band1] / self.FovRefMean1[Band1])
                homoEnv1 = np.abs(self.EnvRefStd1[Band1] / self.EnvRefMean1[Band1])
                homoFovEnv1 = np.abs(self.FovRefMean1[Band1] / self.EnvRefMean1[Band1] - 1)
                homoValue1 = self.FovRefMean1[Band1]
                homoFov2 = np.abs(self.FovRefStd2[Band1] / self.FovRefMean2[Band1])
                homoEnv2 = np.abs(self.EnvRefStd2[Band1] / self.EnvRefMean2[Band1])
                homoFovEnv2 = np.abs(self.FovRefMean2[Band1] / self.EnvRefMean2[Band1] - 1)
                homoValue2 = self.FovRefMean2[Band1]
            #### 云判识关闭状态 ####
            if (modeCfg.clear_min_ir == 0 and 'ir' in flag) or (modeCfg.clear_max_vis == 0 and 'vis' in flag):

                condition = np.logical_and(self.MaskRough > 0, True)
                condition = np.logical_and(SatZRaio < th1, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,角度均匀性过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFov1 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoEnv1 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,环境过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFovEnv1 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区环境过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoValue1 < th_vaue_max, condition)
                condition = np.logical_and(homoValue1 > 0, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,饱和值过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                # sat 2过滤

                condition = np.logical_and(homoFov2 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoEnv2 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,环境2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoFovEnv2 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,靶区环境2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(homoValue2 > 0, condition)
                condition = np.logical_and(homoValue2 < th_vaue_max, condition)
                idx = np.where(condition)
                print u'%s %s 云判识关闭,饱和值2过滤后，精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                self.MaskFine[Band1][idx] = 1

            #### 云判识开启状态 ####
            else:
                # 晴空判别
                if 'ir' in flag:
                    condition = np.logical_and(self.MaskRough > 0, irValue >= modeCfg.clear_min_ir)
                elif 'vis' in flag:
                    condition = np.logical_and(self.MaskRough > 0, visValue < modeCfg.clear_max_vis)
                    condition = np.logical_and(visValue > 0, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(SatZRaio < th1, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 角度 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFov1 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv1 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 环境1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv1 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区/环境 1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue1 > 0, condition)
                condition = np.logical_and(homoValue1 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))


                condition = np.logical_and(homoFov2 < th2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv2 < th3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 环境2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv2 < th4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 靶区/环境 2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue2 > 0, condition)
                condition = np.logical_and(homoValue2 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,晴空点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                idx_clear = np.where(condition)
                self.MaskFine[Band1][idx_clear] = 1

                # 云区判别
                if 'ir' in flag:
                    condition = np.logical_and(self.MaskRough > 0, irValue < modeCfg.clear_min_ir)
                    condition = np.logical_and(irValue > 0, condition)
                    idx = np.where(condition)
                elif 'vis' in flag:
                    condition = np.logical_and(self.MaskRough > 0, visValue >= modeCfg.clear_max_vis)

                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点个数 %d' % (Band1, flag, len(idx[0]))

                condition = np.logical_and(SatZRaio < th_cld1, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 角度 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFov1 < th_cld2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv1 < th_cld3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 环境1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv1 < th_cld4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区/环境 1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue1 > 0, condition)
                condition = np.logical_and(homoValue1 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))


                condition = np.logical_and(homoFov2 < th_cld2, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoEnv2 < th_cld3, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 环境2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoFovEnv2 < th_cld4, condition)
                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 靶区/环境 2 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))
                condition = np.logical_and(homoValue2 > 0, condition)
                condition = np.logical_and(homoValue2 < th_vaue_max, condition)

                idx = np.where(condition)
                print u'%s %s 云判识开启,云区点中 饱和值1 均匀过滤后, 精匹配点个数 %d' % (Band1, flag, len(idx[0]))


                idx_cloud = np.where(condition)
                totalNums = len(idx_cloud[0]) + len(idx_clear[0])
                print u'%s %s 云判识开启，匹配点个数，晴空 %d 云区 %d 总计：%d' % (Band1, flag, len(idx_clear[0]), len(idx_cloud[0]), totalNums)
                self.MaskFine[Band1][idx_cloud] = 1


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
