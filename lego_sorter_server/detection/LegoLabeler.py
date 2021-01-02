class LegoLabeler:

    def to_label_file(self, filename, path, image_width, image_height, bbs_xyxy_array):
        objects = ""
        for coord in bbs_xyxy_array:
            objects += self.get_object(*coord)

        return f"""<annotation>
                <folder>images</folder>
                <filename>{filename}</filename>
                <path>{path}</path>
                <source>
                        <database>LegoSorterPGR</database>
                </source>
                <size>
                        <width>{image_width}</width>
                        <height>{image_height}</height>
                        <depth>3</depth>
                </size>
                <segmented>0</segmented>
                {objects}
        </annotation>"""

    @staticmethod
    def get_object(x1, y1, x2, y2):
        return f"""<object>
                        <name>lego</name>
                        <pose>Unspecified</pose>
                        <truncated>0</truncated>
                        <difficult>0</difficult>
                        <bndbox>
                                <xmin>{int(x1)}</xmin>
                                <ymin>{int(y1)}</ymin>
                                <xmax>{int(x2)}</xmax>
                                <ymax>{int(y2)}</ymax>
                        </bndbox>
                </object>"""
