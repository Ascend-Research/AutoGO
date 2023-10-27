from dataclasses import dataclass
from typing import NewType
import copy


DotMotif = NewType('DotMotif', dict)


@dataclass
class DotMotifList:
    """
    Keep track of all generated DotMotif
    """
    def __init__(self) -> None:
        self.item_list = []

    def add(self, item: DotMotif) -> None:
        # Keep a copy of the input item input the list
        copied_item = copy.deepcopy(item)
        self.item_list.append(copied_item)
    
    def pop(self) -> DotMotif:
        # Pop the last item in the list
        poped_item = self.item_list[-1]
        self.item_list = self.item_list[:-1]
        return poped_item
    
    def query(self, index: int) -> DotMotif:
        # Return the item with the given index in the list
        return self.item_list[index]
