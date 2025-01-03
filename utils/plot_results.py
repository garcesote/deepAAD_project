import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

def plot_clsf_results(spatial_classifier, norm_metrics, labels, eval_window, accuracies, save_path = None):

    # Crear una figura con 3 gráficos en una fila
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    # 1. Gráfico de puntos con frontera de decisión
    for label, color, marker, class_label in zip([0, 1], ['red', 'blue'], ['o', 's'], ['unattended', 'attended']):
        axs[0].scatter(
            norm_metrics[labels == label, 0],
            norm_metrics[labels == label, 1],
            color=color,
            label=class_label,
            alpha=0.7,
            edgecolor='k',
            marker=marker
        )

    # Calcular el espacio para la frontera de decisión
    x_min, x_max = norm_metrics[:, 0].min() - 0.1, norm_metrics[:, 0].max() + 0.1
    y_min, y_max = norm_metrics[:, 1].min() - 0.1, norm_metrics[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Generar predicciones para cada punto del grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    decision_boundary = spatial_classifier.predict(grid)  # Usar el clasificador entrenado
    decision_boundary = decision_boundary.reshape(xx.shape)

    # Contorno de la frontera de decisión
    axs[0].contourf(xx, yy, decision_boundary, levels=[-0.1, 0.5, 1.1], colors=['red', 'blue'], alpha=0.2)

    # Configuración del gráfico
    axs[0].set_title('2D Visualization with LDA Decision Boundary')
    axs[0].set_xlabel('Metric 1 (Correlation)')
    axs[0].set_ylabel('Metric 2 (MAE Diff)')
    axs[0].legend()
    axs[0].grid(True)

    projected_data = spatial_classifier.transform(norm_metrics)

    # 2. Histograma de la proyección LDA
    for label, color, class_label in zip([0, 1], ['red', 'blue'], ['unattended', 'attended']):
        axs[1].hist(
            projected_data[labels == label],
            bins=20,
            alpha=0.6,
            color=color,
            label=class_label
        )
    axs[1].set_title('LDA Projection')
    axs[1].set_xlabel('Discriminant Component')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Gráfico de barras horizontal
    acc_labels = ['Att_acc', 'Unatt_acc', 'Global_acc']

    colors = cm.plasma([0.2, 0.5, 0.8])
    axs[2].barh(acc_labels, accuracies, color=colors, alpha=0.7)
    for i, valor in enumerate(accuracies):
        axs[2].text(valor + 2, i, f'{valor:.2f}%', va='center', fontsize=10)
    axs[2].set_title(f'LDA Accuracies')
    axs[2].set_xlim(0, 100)
    axs[2].grid(axis='x', linestyle='--', alpha=0.6)
    axs[2].axes.get_xaxis().set_visible(False)

    # Ajustar diseño
    plt.tight_layout()
    plt.title(f'Resultados clasificación LDA con ventana de {eval_window//64}s')

    # Mostrar o guardar la gráfica si se incluye o no el argumento save_path
    if save_path is None:
        plt.show()
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{eval_window//64}s.png'))