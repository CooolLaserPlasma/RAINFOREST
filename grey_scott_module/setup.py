from distutils.core import setup, Extension
setup(name = 'hello_python', version = '1.0',  \
   ext_modules = [Extension('myModule', ['test.c'])])