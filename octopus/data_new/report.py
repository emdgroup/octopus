from attrs import define, field


@define
class Report:
    data: list[dict] = field(factory=list)

    def add_column_info(self, column_name: str, info: dict):
        # Find the column info if it exists
        for column in self.data:
            if column_name in column:
                column[column_name].update(info)
                return
        # If column info does not exist, add a new column
        self.data.append({column_name: info})

    def add_section(self, section_name: str, section_data: dict):
        # Check if the section already exists
        for section in self.data:
            if section_name in section:
                section[section_name] = section_data
                return
        # If section does not exist, add a new section
        self.data.append({section_name: section_data})

    def get_column_info(self, column_name: str) -> dict:
        for column in self.data:
            if column_name in column:
                return column[column_name]
        return {}


# class Report:
#     data: dict = field(factory=dict)

#     def add_column_info(self, column_name, info):
#         if column_name not in self.data:
#             self.data[column_name] = {}
#         self.data[column_name].update(info)

#     def add_section(self, section_name, section_data):
#         self.data[section_name] = section_data

#     def get_column_info(self, column_name):
#         return self.data.get(column_name, {})

#     def __str__(self):
#         return str(self.data)
