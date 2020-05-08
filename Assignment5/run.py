import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def poly_model(x, y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()

    model.fit(x_poly, y)
    return poly, model


def poly_preds(x, poly, model):
    transformed = [poly.transform([i]) for i in x]
    return [model.predict(a)[0] for a in transformed]


def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y) ** 2))


def train_models(dataframe,
                 component,
                 target,
                 degrees,
                 output_plot):
    results = dataframe.copy()
    x = dataframe[component].values.reshape(-1, 1)
    y = dataframe[target].values.reshape(-1, 1)
    x_plot = np.linspace(min(x), max(x), 160)

    model = LinearRegression()
    model.fit(x, y)
    print('Coefficient: {}'.format(model.coef_))
    print('Intercept: {}'.format(model.intercept_))

    linear_results = [model.predict([a])[0] for a in x]
    results['final_linear'] = [l[0] for l in linear_results]
    rmse_linear = rmse(y, linear_results)
    print(f'RMSE for linear model: {rmse_linear}')

    plot_linear_results = [model.predict([a])[0] for a in x_plot]
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Training points')
    ax.plot(x_plot, plot_linear_results, color='r', label='linear',
            linewidth=1)

    print('Computing polynomial features and training models...')
    cmap = {0: 'k', 1: 'b', 2: 'y', 3: 'g', 4: 'orange'}

    for d, color in zip(degrees, cmap.values()):
        poly, model = poly_model(x, y, d)
        preds = poly_preds(x_plot, poly, model)
        res_preds = poly_preds(x, poly, model)
        rmse_poly = rmse(y, res_preds)
        print(f'  RMSE for poly degree {d}: {rmse_poly}')

        results[f'final_poly{d}'] = [p[0] for p in res_preds]

        ax.plot(x_plot, preds, color=color, label=f'poly ({d})', linewidth=1,
                linestyle='--')

    ax.set_xlabel(component)
    ax.set_ylabel(target)
    ax.grid()
    ax.set_title(f'{target} on {component}')
    ax.legend()

    fig.savefig(output_plot)

    return results


pathlib.Path('./output').mkdir(parents=True, exist_ok=True)
df = pd.read_csv('./specs/markB_question.csv')

print('*' * 20)
print('Part 1: MCQ1')
print('*' * 20)

mcq1 = train_models(df, component='MCQ1', target='final',
                    degrees=[2, 3, 4, 8, 10],
                    output_plot='./output/question_mcq1.pdf')

print()
print('*' * 20)
print('Part 2: MCQ2')
print('*' * 20)
mcq2 = train_models(df, component='MCQ2', target='final',
                    degrees=[2, 3, 4, 8, 10],
                    output_plot='./output/question_mcq2.pdf')

mcq1.to_csv('./output/question_mcq1.csv', index=False)
mcq2.to_csv('./output/question_mcq2.csv', index=False)

print()
print('*' * 20)
print('Part 3: Full model')
print('*' * 20)

x_full = df[['MCQ1', 'MCQ2']].values.reshape(-1, 2)
y = df['final'].values.reshape(-1, 1)
results = df.copy()

full_model = LinearRegression()
full_model.fit(df[['MCQ1', 'MCQ2']], y)

print('Coefficient: {}'.format(full_model.coef_))
print('Intercept: {}'.format(full_model.intercept_))

lin_res = [full_model.predict([a]) for a in x_full]
results['final_linear'] = [l[0][0] for l in lin_res]
rmse_linear = rmse(y, lin_res)
print(f'RMSE for full linear model: {rmse_linear}')

results.to_csv('./output/question_full.csv', index=False)
