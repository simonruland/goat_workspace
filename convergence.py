import itertools as it
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


colors = {
    'white': '#ffffff',
    'black': '#000000',
    'orange': '#e69f00',
    'skyblue': '#56b4e9',
    'bluishgreen': '#009e73',
    'yellow': '#f0e442',
    'blue': '#0072b2',
    'vermillion': '#d55e00',
    'reddishpurple': '#cc79a7'
}


def main(version, association, cutoff, plot_cutoff, load, compare_with_final):

    if not load:
        out = pd.read_csv('./results/{0}_out_{1}-{2}.csv'.format(version, association, cutoff))
        out.columns = out.columns.astype(int)

        n = out.shape[1]
        m = out.shape[0]
        arr = out.to_numpy() - 1
        rows = arr.flatten()
        cols = np.tile(np.arange(n), len(arr))
        mu = np.zeros((n, n), dtype=int)
        np.add.at(mu, (rows, cols), 1)
        
        g_mu = nx.DiGraph()
        g_mu.add_nodes_from(range(n))
        for i, j in it.permutations(range(n), r=2):
            if all(sum(mu[i][:r]) >= sum(mu[j][:r]) for r in range(n)):
                g_mu.add_edge(i, j)
        g_mu = nx.transitive_reduction(g_mu)

        if plot_cutoff is not None:
            subgraph = set()
            for dist, nodes in enumerate(nx.topological_generations(g_mu)):
                if dist <= plot_cutoff:
                    subgraph = subgraph.union(nodes)
            g_mu = nx.subgraph(g_mu, subgraph)
        else:
            subgraph = g_mu.nodes()

        g_mu_adj = nx.to_numpy_array(g_mu)
        
        data = []
        nu = np.zeros((n, n))
        g_nu_adj = np.zeros((len(subgraph), len(subgraph)))
        ct = 0
        for row in arr:
            ct += 1
            for r in range(n):
                nu[row[r], r] += 1
            if ct % 1000 == 0:
                g_nu_prior_adj = g_nu_adj
                try:
                    g_nu = nx.DiGraph()
                    g_nu.add_nodes_from(range(n))
                    for i, j in it.permutations(range(n), r=2):
                        if all(sum(nu[i][:r]) >= sum(nu[j][:r]) for r in range(n)):
                            g_nu.add_edge(i, j)
                    g_nu = nx.subgraph(g_nu, subgraph)
                    g_nu = nx.transitive_reduction(g_nu)
                    g_nu_adj = nx.to_numpy_array(g_nu)
                    if compare_with_final:
                        norm = np.linalg.norm(g_mu_adj - g_nu_adj, ord='fro')
                    else:
                        norm = np.linalg.norm(g_nu_adj - g_nu_prior_adj, ord='fro')
                    print('sample: {0}, norm: {1}'.format(ct, norm))
                    data.append((ct, norm))
                except (nx.NetworkXError, AttributeError):
                    continue

                df = pd.DataFrame(data, columns=['sample', 'norm'])
                df.to_csv('./convergence/convergence_{0}-{1}-{2}-{3}.csv'.format(
                    version, association, cutoff, plot_cutoff), index=False
                )

    else:

        df = pd.read_csv('./convergence/convergence_{0}-{1}-{2}-{3}.csv'.format(
            version, association, cutoff, plot_cutoff)
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['sample'],
        y=df['norm'],
        mode='markers+lines',
        marker=dict(
            color=colors['orange'] if association == 'atp' else colors['blue'],
            symbol='circle' if association == 'atp' else 'square'
        ),
    ))
    fig.update_layout(
        xaxis_title='Number of Samples',
        yaxis_title='Norm',
        margin=dict(b=10, l=10, r=10, t=10),
    )

    fig.write_image(
        './convergence/convergence_{0}-{1}-{2}-{3}.pdf'.format(version, association, cutoff, plot_cutoff),
        width=1200, height=325, scale=1
    )


if __name__ == '__main__':

    ver_l = ['nonadj']
    assc_l = ['wta', 'atp']
    ctff_l = [3, 5]
    plt_ctff_l = [5, None]
    ld = False
    final = False

    for ver in ver_l:
        for assc in assc_l:
            for ctff in ctff_l:
                for plt_ctff in plt_ctff_l:
                    main(ver, assc, ctff, plt_ctff, ld, final)
