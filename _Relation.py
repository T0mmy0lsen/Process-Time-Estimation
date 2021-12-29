import sql
from _Object import Object


class Relation:

    _object = None

    def __init__(self, el):
        self.id = el[0]
        self.leftId = el[1]
        self.rightId = el[2]
        self.leftType = el[3]
        self.rightType = el[4]

    def get_right(self):
        query = "SELECT `id`, `name`, `objectType` FROM `object_history` WHERE `id` = %s"
        el = sql.SQL().one(query)
        self._object.append(Object(el))
        return self
