from tqdm import tqdm

class TqdmWrapper(tqdm):
    """提供了一個 `total_time` 格式參數"""
    @property
    def format_dict(self):
        # 取得父類別的format_dict
        d = super().format_dict
        
        # 計算總共花費的時間
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        
        # 更新字典以包含總共花費的時間
        d.update(total_time='總計: ' + self.format_interval(total_time))

        # 返回更新後的字典
        return d