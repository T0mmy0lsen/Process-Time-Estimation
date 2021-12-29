import sql
from _Relation import Relation


class Request:

    _relations = []

    def __init__(self):
        self.id = 0

    def get_relations(self):
        query = "SELECT `id`, `leftId`, `leftType`, `rightId`, `rightType` FROM `relation_history` WHERE `leftId` = %s"
        for el in sql.SQL().all(query):
            self._relations.append(Relation(el))
        return self
