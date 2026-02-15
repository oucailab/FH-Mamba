
# 通过调用 write_netcdf 函数并将上一步的输出文件名data.txt作为参数传递，读取并存储处理过的数据，以方便下次读取数据
from ..utils import write_netcdf

start_time = 19910101
end_time = 20201231
write_netcdf("data.txt", start_time, end_time, "full_sic.nc")
