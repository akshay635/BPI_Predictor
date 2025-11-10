import pandas as pd

def clean_data(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["season"] = df["season"].astype(int)
    df['batting_average'] = np.where(df['dismissals'] > 0, round((df['total_runs'] / df['dismissals']), 2), 0)
    df['boundaries(4s/6s)'] = df['fours'].astype(str) + '/' + df['sixes'].astype(str)
    df['boundaries_runs'] = df['fours']*4 + df['sixes']*6
    df['runs/ball'] = round((df['total_runs'] / df['total_balls']), 2)
    df['CI(%)'] = round(((df['thirty_plus']*1 + df['fifty_plus']*2 + df['hundred_plus']*3) / df['matches'])*100, 2)
    df['runs_rbw'] = df['total_runs'] - df['boundaries_runs']
    df['runs_rbw%'] = round((df['runs_rbw'] / df['total_runs'])*100, 2)
    df['boundary_runs%'] = round((df['boundaries_runs'] / df['total_runs'])*100, 2)
    df['dot_balls%'] = round((df['dot_balls']/df['total_balls'])*100, 2)
    df['batting_quality'] = round(((df['batting_average']/30) + (df['strike_rate']/130))/2, 2)
    df['frequency'] = round((df['total_balls']/df['matches']), 2)
    df['BPI_Score'] = round((df['batting_quality'] * df['frequency']), 2)
    df['Batting_Index'] = round((df['strike_rate'] * df['batting_average'])/100, 2)

    return df
    