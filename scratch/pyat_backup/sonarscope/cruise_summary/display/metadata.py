from sonar_netcdf.utils.print_color import info

from pyat.sonarscope.cruise_summary.global_data import GlobalDataModel


class Metadata:
    def __init__(self, data: GlobalDataModel):
        self.data = data

    def print_global_metadata(self):
        # print global attributes
        for k, v in self.data.metadata.attributes.items():
            info(f"{k}: {v}")
        info("")
        info(f"Number of files : {self.data.metadata.file_count}")
        info(f"Number of ping : {self.data.metadata.ping_count}")
        info("")
        if self.data.metadata.ping_count > 0:

            info(f"Date-time range : {self.data.get_min_date()} to {self.data.get_max_date()}")
            min_latitude = self.data.get_min_latitude()
            max_latitude = self.data.get_max_latitude()
            min_longitude = self.data.get_min_longitude()
            max_longitude = self.data.get_max_longitude()
            info(
                f"Medium latitude : {(max_latitude + min_latitude) / 2:.3f} ({min_latitude:.3f} to {max_latitude:.3f})"
            )
            info(
                f"Mean longitude : {(max_longitude + min_longitude) / 2:.3f} ({min_longitude:.3f} to {max_longitude:.3f})"
            )
            info(f"\nVariables statistics")
            for name, value in self.data.metadata.variable_metadata.items():
                info(f"\t{name} : min = {value.min_value} max ={value.max_value}")
