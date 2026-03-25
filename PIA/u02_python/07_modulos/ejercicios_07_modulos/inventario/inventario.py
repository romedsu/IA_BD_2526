import productos
import ventas
#import utilidades

articulo=input('Introduce el artículo: ')

cantidad= int(input('Introduce la cantidad: '))


ventas.gestion(articulo,cantidad,productos.productos)

#utilidades.total(articulo,cantidad,productos.productos)

