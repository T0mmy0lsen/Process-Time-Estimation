import sql
from _Object import Object


class Relation:

    _object = None

    def __init__(self, el):
        self.id = el['id']
        self.leftId = el['leftId']
        self.rightId = el['rightId']
        self.leftType = el['leftType']
        self.rightType = el['rightType']

    def get_right(self):
        query = "SELECT `id`, `name`, `objectType` FROM `object_history` WHERE `id` = %s"
        el = sql.SQL().one(query)
        self._object.append(Object(el))
        return self
