from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt


class ListItem(QtWidgets.QListWidgetItem):
    def __init__(self, label):
        super(ListItem, self).__init__(label)
        self.__hasCheckbox = False

    @property
    def hasCheckbox(self):
        return self.__hasCheckbox

    def setUserData(self, data):
        self.setData(Qt.ItemDataRole.UserRole, data)

    def getUserData(self):
        return self.data(Qt.ItemDataRole.UserRole)

    def text(self):
        return str(super(ListItem, self).text())

    def enableCheckbox(self):
        self.__hasCheckbox = True
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.checkedState = False

    def setChecked(self, checked):
        if not self.hasCheckbox:
            self.setSelected(checked)
            return

        if checked:
            self.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.checkedState = self.checkState()

    def isChecked(self):
        return self.hasCheckbox and self.checkState() != QtCore.Qt.CheckState.Unchecked

    def _clicked(self):
        # owner = self.listWidget()
        if self.hasCheckbox:
            if self.checkState() != self.checkedState:
                self.checkedState = self.checkState()
                if self.checkState():
                    print("Item checked")
                else:
                    print("Item unchecked")
                return True
        return False


class ListView(QtWidgets.QListWidget):
    def __init__(self):
        super(ListView, self).__init__()
        # Widget.__init__(self)
        self._vertical_scrolling = True
        self.itemActivated[QtWidgets.QListWidgetItem].connect(self._activate)
        self.itemClicked[QtWidgets.QListWidgetItem].connect(self._clicked)

    def _activate(self, item):
        print("Item activated")

    def _clicked(self, item):
        if item._clicked():
            return
        if self.allowsMultipleSelection():
            if item.isSelected():
                print("Item selected")
            else:
                print("Item deselected")
        else:
            pass

    def onActivate(self, event):
        pass

    def onClicked(self, event):
        pass

    def setData(self, items):
        self.clear()
        for item in items:
            self.addItem(item)

    def setVerticalScrollingEnabled(self, enabled):
        self._vertical_scrolling = enabled
        if enabled:
            self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.updateGeometry()

    def rowCount(self):
        return len([item for item in self.getItems() if item is not None and not item.isHidden()])

    def sizeHint(self):
        if self._vertical_scrolling:
            return super(ListView, self).sizeHint()
        else:
            rows = self.rowCount()
            if rows > 0:
                rowHeight = self.sizeHintForRow(0)
                rowHeight = max(rowHeight, self.iconSize().height())
            else:
                rowHeight = 0
            height = rowHeight * rows

            size = super(ListView, self).sizeHint()
            size.setHeight(height)
            return size

    _brushes = {}

    @classmethod
    def getBrush(cls, color):
        if color not in cls._brushes:
            cls._brushes[color] = QtGui.QBrush(QtGui.QColor(color))
        return cls._brushes[color]

    def addLogItem(self, text, color=None, data=None, checkbox=False, pos=None):
        item = ListItem(text)
        item.setText(text)
        if color is not None:
            item.setForeground(self.getBrush(color))
        if data is not None:
            item.setUserData(data)
        if checkbox:
            item.enableCheckbox()
        return self.addItemObject(item, pos)

    def addItemObject(self, item, pos=None):
        if pos is not None:
            super(ListView, self).insertItem(pos, item)
        else:
            super(ListView, self).addItem(item)
        if not self._vertical_scrolling:
            self.updateGeometry()
        return item

    def getSelectedItem(self):
        items = self.selectedItems()
        if len(items) > 0:
            return items[0].text
        return None

    def getSelectedItems(self):
        return [item.text for item in self.selectedItems()]

    def getItemData(self, row):
        item = self.item(row)
        if item is not None:
            return item.getUserData()

    def setItemColor(self, row, color):
        item = self.item(row)
        if item is not None:
            item.setForeground(self.getBrush(color))

    def showItem(self, row, state):
        item = self.item(row)
        if item is not None:
            item.setHidden(not state)

    def allowMultipleSelection(self, allow):
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
            if allow
            else QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )

    def allowsMultipleSelection(self):
        return self.selectionMode() == QtWidgets.QAbstractItemView.SelectionMode.MultiSelection

    def getItems(self):
        return [self.item(row) for row in range(self.count())]

    def clearSelection(self):
        super(ListView, self).clearSelection()

        for item in self.getItems():
            if item is None:
                continue
            if item.isSelected():
                item.setSelected(False)

        self.callEvent("onClearSelection", None)
