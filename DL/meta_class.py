# !/usr/bin/python
import threading
class Field(object):
    def __init__(self, name, column_type):
        self.name = name
        self.column_type = column_type

    def __str__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.name)


class StringField(Field):
    def __init__(self, name):
        super(StringField, self).__init__(name, 'varchar(100')


class IntegerField(Field):
    def __init__(self, name):
        super(IntegerField, self).__init__(name,'bigint')


class ModelMetaclass(type):
    def __new__(cls, name, bases, attrs):
        if name=='Model':
            return type.__new__(cls, name, bases, attrs)
        print("Found model: %s." % name)
        mapping = dict()
        for k, v in attrs.items():
            if isinstance(v, Field):
                print("Found mapping: %s ==> %s." % (k,v))
                mapping[k] = v
        for k in mapping.keys():
            attrs.pop(k)
        attrs['__mappings__'] = mapping
        attrs['__table__'] = name
        return type.__new__(cls, name, bases, attrs)


class Model(dict, metaclass=ModelMetaclass):
    def __init__(self, **kw):
        super(Model, self).__init__(**kw)
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(r'Model object has no attribute %s' % item)

    def __setattr__(self, key, value):
        self[key] = value

    def save(self):
        fields = []
        paras = []
        args = []
        for k, v in self.__mappings__.items():
            fields.append(v.name)
            paras.append('?')
            args.append(getattr(self, k, None))
        sql = 'insert into %s (%s) values (%s)' % (self.__table__, ','.join(fields), ','.join(paras))
        print('SQL %s' % sql)
        print('QRGS: %s' % str(args))



class User(Model):
    id = IntegerField('id')
    name = StringField('name')

u = User(id=12345, name='Michael')
u.save()
print(u.__mappings__)