import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# 1. GENERACIÓN DE DATOS FICTICIOS
# ==========================================

np.random.seed(42)

# Generar fechas para un año
fecha_inicio = datetime(2023, 1, 1)
fechas = [fecha_inicio + timedelta(days=x) for x in range(365)]

# Productos y categorías
productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares', 
             'Webcam', 'USB', 'Disco Duro', 'Memoria RAM', 'Router']
categorias = ['Computadoras', 'Accesorios', 'Accesorios', 'Computadoras', 'Accesorios',
              'Accesorios', 'Accesorios', 'Almacenamiento', 'Componentes', 'Redes']
precios_base = [800, 15, 45, 250, 60, 80, 10, 120, 90, 55]

# Generar transacciones (entre 5 y 20 por día)
datos_ventas = []
for fecha in fechas:
    num_transacciones = np.random.randint(5, 21)
    for _ in range(num_transacciones):
        idx_producto = np.random.randint(0, len(productos))
        cantidad = np.random.randint(1, 6)
        # Agregar variación al precio (+/- 10%)
        precio_unitario = precios_base[idx_producto] * np.random.uniform(0.9, 1.1)
        total = precio_unitario * cantidad
        
        datos_ventas.append({
            'fecha': fecha,
            'producto': productos[idx_producto],
            'categoria': categorias[idx_producto],
            'cantidad': cantidad,
            'precio_unitario': round(precio_unitario, 2),
            'total': round(total, 2)
        })

# Crear DataFrame
df = pd.DataFrame(datos_ventas)

# Añadir columnas adicionales para análisis
df['mes'] = df['fecha'].dt.month
df['mes_nombre'] = df['fecha'].dt.strftime('%B')
df['dia_semana'] = df['fecha'].dt.day_name()
df['trimestre'] = df['fecha'].dt.quarter

# Guardar a CSV
df.to_csv('ventas_2023.csv', index=False)
print("✓ Datos generados y guardados en 'ventas_2023.csv'")
print(f"Total de transacciones: {len(df)}")

# ==========================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS
# ==========================================

print("\n" + "="*50)
print("ANÁLISIS DE VENTAS - AÑO 2023")
print("="*50)

# Métricas generales
print("\n📊 MÉTRICAS GENERALES:")
print(f"Ingresos totales: ${df['total'].sum():,.2f}")
print(f"Ticket promedio: ${df['total'].mean():,.2f}")
print(f"Unidades vendidas: {df['cantidad'].sum():,}")
print(f"Transacciones totales: {len(df):,}")

# Producto más vendido
print("\n🏆 TOP 5 PRODUCTOS POR INGRESOS:")
top_productos = df.groupby('producto')['total'].sum().sort_values(ascending=False).head(5)
for i, (producto, total) in enumerate(top_productos.items(), 1):
    print(f"{i}. {producto}: ${total:,.2f}")

# Análisis por categoría
print("\n📦 VENTAS POR CATEGORÍA:")
ventas_categoria = df.groupby('categoria')['total'].sum().sort_values(ascending=False)
for categoria, total in ventas_categoria.items():
    porcentaje = (total / df['total'].sum()) * 100
    print(f"{categoria}: ${total:,.2f} ({porcentaje:.1f}%)")

# Mejor mes
print("\n📅 MEJOR MES DEL AÑO:")
ventas_mensuales = df.groupby('mes_nombre')['total'].sum().sort_values(ascending=False)
mejor_mes = ventas_mensuales.index[0]
mejor_mes_ventas = ventas_mensuales.iloc[0]
print(f"{mejor_mes}: ${mejor_mes_ventas:,.2f}")

# Día de la semana más rentable
print("\n📆 VENTAS POR DÍA DE LA SEMANA:")
ventas_dia = df.groupby('dia_semana')['total'].sum().sort_values(ascending=False)
for dia, total in ventas_dia.items():
    print(f"{dia}: ${total:,.2f}")

# ==========================================
# 3. VISUALIZACIONES
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Dashboard de Análisis de Ventas 2023', fontsize=16, fontweight='bold')

# Gráfico 1: Ventas mensuales
ventas_mes = df.groupby('mes')['total'].sum()
axes[0, 0].plot(ventas_mes.index, ventas_mes.values, marker='o', linewidth=2, markersize=8)
axes[0, 0].set_title('Ventas Mensuales', fontweight='bold')
axes[0, 0].set_xlabel('Mes')
axes[0, 0].set_ylabel('Ingresos ($)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].ticklabel_format(style='plain', axis='y')

# Gráfico 2: Top 5 productos
top5 = df.groupby('producto')['total'].sum().sort_values(ascending=False).head(5)
axes[0, 1].barh(range(len(top5)), top5.values, color='steelblue')
axes[0, 1].set_yticks(range(len(top5)))
axes[0, 1].set_yticklabels(top5.index)
axes[0, 1].set_title('Top 5 Productos por Ingresos', fontweight='bold')
axes[0, 1].set_xlabel('Ingresos ($)')
axes[0, 1].invert_yaxis()

# Gráfico 3: Ventas por categoría (pie chart)
ventas_cat = df.groupby('categoria')['total'].sum()
axes[1, 0].pie(ventas_cat.values, labels=ventas_cat.index, autopct='%1.1f%%', startangle=90)
axes[1, 0].set_title('Distribución por Categoría', fontweight='bold')

# Gráfico 4: Tendencia semanal
ventas_diarias = df.groupby('fecha')['total'].sum()
axes[1, 1].plot(ventas_diarias.index, ventas_diarias.values, alpha=0.6, linewidth=1)
# Media móvil de 7 días
ventas_diarias_ma = ventas_diarias.rolling(window=7).mean()
axes[1, 1].plot(ventas_diarias_ma.index, ventas_diarias_ma.values, 
                color='red', linewidth=2, label='Media Móvil 7 días')
axes[1, 1].set_title('Tendencia de Ventas Diarias', fontweight='bold')
axes[1, 1].set_xlabel('Fecha')
axes[1, 1].set_ylabel('Ingresos ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_ventas_2023.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizaciones guardadas en 'analisis_ventas_2023.png'")

plt.show()

# ==========================================
# 4. INSIGHTS Y RECOMENDACIONES
# ==========================================

print("\n" + "="*50)
print("💡 INSIGHTS Y RECOMENDACIONES:")
print("="*50)

# Tendencia general
ventas_q1 = df[df['trimestre'] == 1]['total'].sum()
ventas_q4 = df[df['trimestre'] == 4]['total'].sum()
crecimiento = ((ventas_q4 - ventas_q1) / ventas_q1) * 100

print(f"\n1. Crecimiento Q1 vs Q4: {crecimiento:+.1f}%")

# Productos de bajo rendimiento
print("\n2. Productos con menores ventas (oportunidad de mejora):")
bottom3 = df.groupby('producto')['total'].sum().sort_values().head(3)
for producto, total in bottom3.items():
    print(f"   - {producto}: ${total:,.2f}")

# Día óptimo
dia_optimo = ventas_dia.index[0]
print(f"\n3. Día óptimo para promociones: {dia_optimo}")

print(f"\n4. Producto estrella: {top_productos.index[0]} genera el {(top_productos.iloc[0]/df['total'].sum()*100):.1f}% de ingresos")

print("\n" + "="*50)
print("✓ Análisis completado exitosamente")
print("="*50)