import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def old_fn():
    # Parameters setup
    #a_m_values = [0.7, 0.8, 0.9, 0.99, 0.999, 0.9999]
    a_m_values = [0.7, 0.8, 0.9, 0.99, 0.999]
    print(a_m_values)
    c_d_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]           # Rows: Different values for C_d
    c_f_list = [1.5, 10, 100, 1000, 10000, 100000]           # Columns: Different values for C_f

    fig, axes = plt.subplots(len(c_d_list), len(c_f_list), figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(r'Sensitivity of $C_{max\_gain}$ to $a_m$ across $C_d$ and $C_f$', fontsize=16)

    gain_values = []

    for i, cd in enumerate(c_d_list):
        for j, cf in enumerate(c_f_list):
            # Calculate C_max_gain based on the formula
            c_max_gain = [1.0 - cd - cf * (1 - a_m) for a_m in a_m_values]
            for value in c_max_gain:
                gain_values.append([cd, cf, value])
            
            ax = axes[i, j]
            #set log scale for x axis
            ax.set_xscale('log')
            ##set y axis to be 0 to 1
            ax.set_ylim(0, 1)
            ax.plot(a_m_values, c_max_gain, color='teal', linewidth=2)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Formatting titles for clarity
            if i == 0:
                ax.set_title(f'$C_f = {cf}$')
            if j == 0:
                ax.set_ylabel(f'$C_d = {cd}$\n$C_{{max\\_gain}}$')
            if i == len(c_d_list) - 1:
                ax.set_xlabel('$a_m$ (Model Accuracy)')


    gain_df = pd.DataFrame(gain_values, columns=['cd', 'cf', 'gain'])
    #find max gian for each cf
    max_gain_df = gain_df.groupby('cf').agg({'gain': 'max'})
    print(max_gain_df)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def calculate_c_full(am, ad, pdm, cd, cf):
    """
    Calculates the full cost based on the derived Equation 6.
    """
    term1 = cd * (1 - am - ad + 2 * pdm)
    term2 = cf * (1 - am - ad + pdm)
    term3 = ad + am - 2 * pdm
    return term1 + term2 + term3




def parameter_sweep_on_costs():
    # Define the parameter ranges for the sweep
    # am (model accuracy) and ad (decision accuracy) range from 0 to 1
    #ad_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    #am_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    ad_values = [0.999]
    am_values = [0.999]
    #c_d_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]           # Rows: Different values for C_d
    c_d_list = [0.00001]
    #c_f_list = [1.5, 3, 5, 10, 20, 100, 1000, 3000, 5000, 10000, 100000]           # Columns: Different values for C_f
    c_f_list = [100000]           # Columns: Different values for C_f


    results = []

    for ad in ad_values:
        for am in am_values:
            for cd in c_d_list:
                for cf in c_f_list:
                    # Constraint: Pdm is the prob that both are correct. 
                    # For a simple sweep, we assume Pdm = am * ad (independence)
                    # or follow your 'perfect' vs 'random' logic.
                    min_pdm = max(0, (am + ad - 1))
                    max_pdm = min(am, ad)
                    step = (max_pdm - min_pdm) / 10
                    if step < 0:
                        raise ValueError("Step is negative")
                    pdm_values = [min_pdm + x * step for x in range(10)] + [am * ad]
                    #pdm_values = [am * ad]
                    
                    for pdm in pdm_values:
                        cost = calculate_c_full(am, ad, pdm, cd, cf)    
                        saving = 1 - cost
                        results.append({"am": am, "ad": ad, "cd": cd, "cf": cf, "pdm": pdm, "saving": saving})
                        print({"am": am, "ad": ad, "cd": cd, "cf": cf, "pdm": pdm, "saving": saving, "cost": cost})

    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(results)

    #find max gian for each cf
    #max_gain_df = df.groupby('cf').agg({'saving': 'max'})
    #print(max_gain_df)


    # Visualization
    #plt.figure(figsize=(10, 6))
    #for ad in ad_values:
    #    subset = df[df['ad'] == ad]
    #    plt.plot(subset['am'], subset['saving'], marker='o', label=f'Decision Accuracy (ad) = {ad}')

    #plot scatter plot of distribution of savings against cf with cf as x axis and saving as y axis
    #plt.figure(figsize=(10, 6))
    #plt.scatter(df['cf'], df['saving'])
    #plt.xscale('log')
    #plt.ylim(0, 1)
    #plt.xlabel('C_f')
    #plt.ylabel('Saving')
    #plt.title('Distribution of savings against C_f')
    #plt.show()


    df["am_ad_sum_by_2"] = (df['am'] + df['ad']) / 2
    #df["min_am_ad"] = df['am'] + df['ad']
    df = df[df['saving'] >= 0.9]

    #group by cf and find the min of min_am_ad and find am and ad values in the same row
    #min_am_ad_df = df.groupby('cf').apply( lambda x: x[x['am_ad_sum_by_2'] == x['am_ad_sum_by_2'].min()])
    #print(min_am_ad_df)

    #print the mean of the min_am_ad for each cf
    print(df.groupby('cf').agg({'am_ad_sum_by_2': 'min'}))


    #plot scatter plot of distribution of cf vs am and color them differently if saving > 0.7 
    #plt.figure(figsize=(10, 6))
    #plt.scatter(df['cf'], df['am'], c=df['saving'] > 0.7, cmap='viridis')
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0, 1)
    #plt.xlabel('C_f')
    #plt.ylabel('Model Accuracy (am)')
    #plt.title('Distribution of cf vs am with color coding for saving > 0.7')
    #plt.show()


def draw_scatter_plots(df):
    # Create a single plot with multiple scatter subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Saving vs Different Parameters', fontsize=16)
    
    # 1. Plot saving vs am
    axes[0, 0].scatter(df['am'], df['saving'], alpha=0.5, s=10)
    axes[0, 0].set_xlabel('Model Accuracy (am)')
    axes[0, 0].set_ylabel('Saving')
    axes[0, 0].set_title('Saving vs Model Accuracy (am)')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Plot saving vs ad
    axes[0, 1].scatter(df['ad'], df['saving'], alpha=0.5, s=10)
    axes[0, 1].set_xlabel('Decision Accuracy (ad)')
    axes[0, 1].set_ylabel('Saving')
    axes[0, 1].set_title('Saving vs Decision Accuracy (ad)')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Plot saving vs cd
    axes[1, 0].scatter(df['cd'], df['saving'], alpha=0.5, s=10)
    axes[1, 0].set_xlabel('C_d')
    axes[1, 0].set_ylabel('Saving')
    axes[1, 0].set_title('Saving vs C_d')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Plot saving vs cf
    axes[1, 1].scatter(df['cf'], df['saving'], alpha=0.5, s=10)
    axes[1, 1].set_xlabel('C_f')
    axes[1, 1].set_ylabel('Saving')
    axes[1, 1].set_title('Saving vs C_f')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temp/saving_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_making_cf_zero(cd=0.00001, cf=100000):
    ad_list = [x/100 for x in range(1, 100, 1)] 
    am_list = [x/100 for x in range(1, 100, 1)] 
    for ad in ad_list:
        for am in am_list:
            k = (am + ad - 1)/ad
            if k < 0 or k > 1:
                print(f"am: {am},{ad} k out of range, cost is higher than 1")
                continue
            cost = calculate_c_full_v2(am, ad, k, cd, cf)
            if cost > 1:
                print(f"am: {am},{ad} cost is higher than 1, {cost}")
            else:
                print(f"am: {am}.{ad} cost is {cost}")

        

def calculate_c_full_v2(am, ad, k, cd, cf):
    """
    Calculates the full cost based on the derived Equation 6.
    """
    #ad + am − 2.ad.k + Cd.(1 − ad − am + 2ad.k) + Cf (1 − ad − am + ad.k)
    constant_term = ad + am - 2*ad*k
    ad_term = cd * (1 - ad - am + 2*ad*k)
    cf_term = cf * (1 - ad - am + ad*k)
    #if cf_term < -0.001:
    #    raise ValueError(f"cf_term is negative: {cf_term}, ad: {ad}, am: {am}, k: {k}")
    if cf_term < 0:
        cf_term = 0

    cost = constant_term + ad_term + cf_term
    #if cf == 100000 and cost < 1:
    #    print(f"### cf: {cf}, cf_term: {cf_term}, cost: {cost}, am: {am}, ad: {ad}, k: {k}")

    return cost

def find_k_values(am, ad):
    k_min = max(0, (am + ad - 1)/ad) 
    k_max = min(1, am/ad)
    if k_max == k_min:
        k_values = [k_min]
    else:
        k_step = (k_max - k_min)/10
        k_values = [k_min + x*k_step for x in range(0, 10)]
    return k_values

def parameter_sweep_on_costs_v2():
    # Define the parameter ranges for the sweep
    # am (model accuracy) and ad (decision accuracy) range from 0 to 1
    
    ad_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    am_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    c_d_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]           # Rows: Different values for C_d
    c_f_list = [1.5, 3, 5, 10, 20, 100, 1000, 3000, 5000, 10000, 100000]           # Columns: Different values for C_f
    
    
    #ad_values = [0.999]
    #am_values = [0.999]
    #c_d_list = [0.001]
    #c_f_list = [100000]           # Columns: Different values for C_f

    results = []

    for ad in ad_values:
        for am in am_values:
            for cd in c_d_list:
                for cf in c_f_list:
                    k_values = find_k_values(am, ad)
                    for k in k_values:
                        cost = calculate_c_full_v2(am, ad, k, cd, cf)
                        cost_saving_factor = 1/cost
                        results.append({"am": am, "ad": ad, "cd": cd, "cf": cf, "k": k, "saving": cost_saving_factor})
                        if cost_saving_factor > 1:
                            k_string = f"{k:.3f}[{k_min:.3f}, {k_max:.3f}]"
                            print({"am": am, "ad": ad, "cd": cd, "cf": cf, "k": k_string, "saving": f"{cost_saving_factor:.3f}", "cost": f"{cost:.2f}"})

    df = pd.DataFrame(results)
    df["am_ad_sum_by_2"] = (df['am'] + df['ad']) / 2

    df = df[df['saving'] >= 1]
    draw_scatter_plots(df)
    
    df = df[df['saving'] >= 9]

    #print(df.groupby('cf').agg({'am_ad_sum_by_2': 'min'}))
    #group by  cf and print the first row with min value am_ad_sum_by_2 for each cf
    print(df.groupby('cf').apply(lambda x: x.loc[x['am_ad_sum_by_2'].idxmin()]))
    #print(df.groupby('cf').apply(lambda x: x[x['am_ad_sum_by_2'] == x['am_ad_sum_by_2'].min()].first()))

def find_max_cf_each_bracket(df, model_accuracy_limit):
    df = df[df['am'] <= model_accuracy_limit]
    df = df[df['ad'] <= model_accuracy_limit]
    #df = df[df['am'] > model_accuracy_limit -5]
    #df = df[df['ad'] > model_accuracy_limit - 5]

    result = df.groupby('cf').agg(
        min_csf=('cost_saving_factor', 'min'),
        avg_csf=('cost_saving_factor', 'mean'),
        p99_csf=('cost_saving_factor', lambda s: s.quantile(0.99)),
        stddev_csf=('cost_saving_factor', 'std'),
        max_csf=('cost_saving_factor', 'max'),
    )
    return result

def parameter_sweep_find_min_configs_for_given_cost():
    cast_saving_factors = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    ad_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    am_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    c_d_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    #c_d_list = [0.00001, 0.000001]
    c_f_list = [100000, 10000, 1000, 100, 10, 5, 3, 1.5] 

    results = []
    cost_saving_factor_index = 0
    for icf,cf in enumerate(c_f_list):
        for iad, ad in enumerate(ad_values):
            for iam, am in enumerate(am_values):
                for icd, cd in enumerate(c_d_list):
                    k_values = find_k_values(am, ad)
                    for k in k_values:
                        cost = calculate_c_full_v2(am, ad, k, cd, cf)
                        cost_saving_factor = 1/cost
                        cost_saving_backet = 10**math.ceil(math.log10(cost_saving_factor))
                        rank = (icf + iad*1.3 + iam*1.3**2 + icd*1.3**3)
                        results.append({"braket": cost_saving_backet, "am": am, "ad": ad, "cd": cd, "cf": cf, "k": k, "cost": cost, 
                            "cost_saving_factor": cost_saving_factor, "rank":rank})
    df = pd.DataFrame(results)
    #print(df)
    #print(df.groupby('braket').apply(lambda x: x.loc[x['rank'].idxmin()]))

    #group by cf, ad, and cd, find the highest cost_saving_factor for each group, sort them by cost_saving_factor in asending order and print
    #reslutsdf = df.groupby(['cf', 'ad', 'cd']).apply(lambda x: x.loc[x['cost_saving_factor'].idxmax()]).sort_values(by='cost_saving_factor', ascending=True)
    #print(reslutsdf)
    #reslutsdf.to_csv('temp/results_v3.csv', index=False)

    #df[df['cost_saving_factor'] >= 45].sort_values(by='cost_saving_factor', ascending=True).to_csv('temp/result_50.csv', index=False)
    # find all data points where cf=100000 am=0.9 ad=0.9 and cost_saving_factor >4 and print
    print(df[(df['cf'] == 100000) & (df['am'] == 0.9) & (df['ad'] == 0.9) & (df['cost_saving_factor'] > 4)])

    print(find_max_cf_each_bracket(df, 0.9))

def parameter_sweep_cost_saving_factor_distribution_v3():
    cast_saving_factors = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    ad_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    am_values = [x/100 for x in range(70, 100, 1)] + [0.999]
    c_d_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    #c_d_list = [0.00001, 0.000001]
    c_f_list = [100000, 10000, 1000, 100, 10, 5, 3, 1.5] 

    results = []
    cost_saving_factor_index = 0
    for icf,cf in enumerate(c_f_list):
        for iad, ad in enumerate(ad_values):
            for iam, am in enumerate(am_values):
                for icd, cd in enumerate(c_d_list):
                    k_values = find_k_values(am, ad)
                    if 2*cd + cf >= 2:
                        k_value_at_min_cost = k_values[0]
                        k_value_at_1percent = k_values[0] + (k_values[-1] - k_values[0]) * 0.01
                        k_value_at_5percent = k_values[0] + (k_values[-1] - k_values[0]) * 0.05
                    else:
                        k_value_at_min_cost = k_values[-1]
                        k_value_at_1percent = k_values[-1] - (k_values[-1] - k_values[0]) * 0.01
                        k_value_at_5percent = k_values[-1] - (k_values[-1] - k_values[0]) * 0.05


                    min_cost = calculate_c_full_v2(am, ad, k_value_at_min_cost, cd, cf)
                    k_1percent_cost = calculate_c_full_v2(am, ad, k_value_at_1percent, cd, cf)
                    k_5percent_cost = calculate_c_full_v2(am, ad, k_value_at_5percent, cd, cf)

                    acc_category = 0
                    if am > 0.95 and ad > 0.95:
                        acc_category = 95
                    elif am > 0.90 or ad > 0.90:
                        acc_category = 90
                    elif am > 0.85 or ad > 0.85:
                        acc_category = 85
                    elif am > 0.80 or ad > 0.80:
                        acc_category = 80
                    elif am > 0.75 or ad > 0.75:
                        acc_category = 75
                    elif am > 0.70 or ad > 0.70:
                        acc_category = 70
                    else:
                        acc_category = 0

                    results.append({"am": am, "ad": ad, "cd": cd, "cf": cf, "acc_category": acc_category, "max_cs": 1/min_cost, 
                        "cs_at_1percent": 1/k_1percent_cost, "cs_at_5percent": 1/k_5percent_cost})


    df = pd.DataFrame(results)
    #group by cf and for each group print min and max values of max_cs
    print("by cf")
    print(df.groupby(['acc_category', 'cf'])['max_cs'].agg(['min', 'max']))
    print(df.groupby(['acc_category', 'cf'])['cs_at_1percent'].agg(['min', 'max']))
    print(df.groupby(['acc_category', 'cf'])['cs_at_5percent'].agg(['min', 'max']))


    results = df.groupby(['acc_category', 'cf']).apply(
        lambda x: f"{x['max_cs'].max():.2f} / {x['cs_at_1percent'].max():.2f}"
    )   

    # 2. Pivot 'acc_category' to columns
    table = results.unstack(level='acc_category')

    print(table)    
    
    #print("by am")

    #print(df.groupby('am')['max_cs'].agg(['min', 'max']))
    #print(df.groupby('am')['cs_at_1percent'].agg(['min', 'max']))
    #print(df.groupby('am')['cs_at_5percent'].agg(['min', 'max']))

    #print("by ad")

    #print(df.groupby('ad')['max_cs'].agg(['min', 'max']))
    #print(df.groupby('ad')['cs_at_1percent'].agg(['min', 'max']))
    #print(df.groupby('ad')['cs_at_5percent'].agg(['min', 'max']))

def human_ai_analyze_emeperical_results():
    df = pd.read_csv('data/ai-human/goemotions-uncertainty.csv')
    print(df.head())
    #groyp by Cf and find min Cost for each group
    results = df.groupby('Cf')['Cost'].agg(['min']).reset_index()
    results["cost_saving_factor"] = 1/results['min']
    

    print(results)
    #plot a line plot of Cf vs min Cost
    plt.plot(results['Cf'], results['cost_saving_factor'])
    plt.xlabel('Cf')
    plt.ylabel('Cost Saving Factor')
    plt.title('Cost Saving Factor vs Cf')
    plt.show()



#parameter_sweep_on_costs()
#parameter_sweep_on_costs_v2()
#parameter_sweep_find_min_configs_for_given_cost()

#print(calculate_c_full_v2(am=0.9, ad=0.9, k=0.888888, cd=0.00001, cf=100000))
#test_making_cf_zero()

#parameter_sweep_cost_saving_factor_distribution_v3()
human_ai_analyze_emeperical_results()
