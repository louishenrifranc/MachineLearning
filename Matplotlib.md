# Basic pour se débrouiller et ploter des graphes

1. Afficher des points  en fonction de X (y et X de même dimension)
    ```python
    plt.scatter(X, y, label='training points', color='lightgray')
    ```
2. Afficher des courbes
    ````python
    plt.plot(range(1, lr.n_iter + 1), lr.cost_)
                                OU
    plt.plot(X_fit, y_lin_fit,
      # les informations du graphes
         label='linear , $R^2=%.2f$' % r2,  
         color='blue',  # couleur
         lw=2,  # largeur du point
         linestyle=':')  # Comment on représente le point
    ```
3. Afficher une légende
    ```python
        plt.xlabel('legend for x axis')
        plt.ylabel('legend for y axis')
        plt.legend(loc='upper right') # ou mettre la légende, cad les "label" de plt.plot
    ```