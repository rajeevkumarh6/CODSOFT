from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLOTS_FOLDER'] = 'static/plots'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PLOTS_FOLDER']):
    os.makedirs(app.config['PLOTS_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('analyze_data', filename=file.filename))
    
    return redirect(request.url)

@app.route('/analyze/<filename>')
def analyze_data(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    sales_data = pd.read_csv(filepath)
    
    # Data cleaning
    sales_data.dropna(inplace=True)
    sales_data['Ship Date'] = pd.to_datetime(sales_data['Ship Date'])
    sales_data.set_index('Ship Date', inplace=True)
    
    # Monthly Sales Trend
    monthly_sales = sales_data.resample('M').sum()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index, monthly_sales['Total Revenue'])
    plt.title('Yearly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plot_filepath = os.path.join(app.config['PLOTS_FOLDER'], 'monthly_sales.png')
    plt.savefig(plot_filepath)
    plt.close()
    
    # Sales by Product Category
    category_sales = sales_data.groupby('Item Type')['Total Revenue'].sum()
    plt.figure(figsize=(10, 6))
    category_sales.plot(kind='bar')
    plt.title('Sales by Product Category')
    plt.xlabel('Item Type')
    plt.ylabel('Total Revenue')
    plot_filepath_category = os.path.join(app.config['PLOTS_FOLDER'], 'category_sales.png')
    plt.savefig(plot_filepath_category)
    plt.close()
    
    # Sales Prediction (Example)
    sales_data['month'] = sales_data.index.month
    X = sales_data[['month']]
    y = sales_data['Total Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred, color='red')
    plt.title('Sales Prediction')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue')
    plot_filepath_prediction = os.path.join(app.config['PLOTS_FOLDER'], 'sales_prediction.png')
    plt.savefig(plot_filepath_prediction)
    plt.close()

    # Pivot Table and Pie Chart for Total Revenue by Country
    pivot_table = pd.pivot_table(
        data=sales_data,
        index='Country',
        values='Total Revenue',
        aggfunc='sum'  # Sum of Total Revenue per Country
    )

    plt.figure(figsize=(12, 5))
    pivot_table.plot(
        kind='pie', 
        y='Total Revenue', 
        autopct='%1.1f%%'  # Show percentages
    )
    plt.ylabel('')  # Hide the y-label
    plt.title('Total Revenue by Country')
    plot_filepath_pie = os.path.join(app.config['PLOTS_FOLDER'], 'revenue_by_country.png')
    plt.savefig(plot_filepath_pie)
    plt.close()
    
    # Line plot for Units Sold vs. Total Cost
    plt.figure()
    sales_data.plot.line(x='Units Sold', y='Total Cost', subplots=True, color={'Total Cost': 'green'})
    plot_filepath_units_vs_cost = os.path.join(app.config['PLOTS_FOLDER'], 'units_vs_cost.png')
    plt.savefig(plot_filepath_units_vs_cost)
    plt.close()

    # Bar plot for Total Revenue, Total Cost, Total Profit
    bar_plot = sales_data.plot.bar(x='Units Sold', y=['Total Revenue', 'Total Cost', 'Total Profit'], color=['purple', 'red', 'blue'], stacked=True)
    plt.xticks(rotation=90)
    plt.locator_params(nbins=38)
    plt.tick_params(axis='y', which='major', labelsize=7)
    plot_filepath_bar = os.path.join(app.config['PLOTS_FOLDER'], 'bar_plot.png')
    plt.savefig(plot_filepath_bar)
    plt.close()

    # Horizontal bar plot for Total Revenue by Item Type
    sales_data.plot.barh(x='Item Type', y='Total Revenue', color='blue')
    plt.locator_params(nbins=28)
    plt.xlabel('Total Revenue')
    plot_filepath_barh = os.path.join(app.config['PLOTS_FOLDER'], 'barh_plot.png')
    plt.savefig(plot_filepath_barh)
    plt.close()

    # Additional Analysis and Charts
    # Most profitable item
    most_profitable_item = sales_data.groupby('Item Type')['Total Profit'].sum().idxmax()
    most_profitable_value = sales_data.groupby('Item Type')['Total Profit'].sum().max()
    
    # Least profitable item
    least_profitable_item = sales_data.groupby('Item Type')['Total Profit'].sum().idxmin()
    least_profitable_value = sales_data.groupby('Item Type')['Total Profit'].sum().min()
    
    # Country with highest sales
    highest_sales_country = sales_data.groupby('Country')['Total Revenue'].sum().idxmax()
    highest_sales_value = sales_data.groupby('Country')['Total Revenue'].sum().max()
    
    # Total profit by most profitable item in a year
    total_profit_per_year = sales_data.groupby(sales_data.index.year)['Total Profit'].sum()
    max_profit_year = total_profit_per_year.idxmax()
    max_profit_value = total_profit_per_year.max()

    # Plot most and least profitable items
    plt.figure(figsize=(10, 6))
    profit_by_item = sales_data.groupby('Item Type')['Total Profit'].sum()
    profit_by_item.plot(kind='bar', color='orange')
    plt.title('Total Profit by Item Type')
    plt.xlabel('Item Type')
    plt.ylabel('Total Profit')
    plot_filepath_profit_item = os.path.join(app.config['PLOTS_FOLDER'], 'profit_by_item.png')
    plt.savefig(plot_filepath_profit_item)
    plt.close()

    # Plot total profit per year
    plt.figure(figsize=(10, 6))
    total_profit_per_year.plot(kind='bar', color='green')
    plt.title('Total Profit per Year')
    plt.xlabel('Year')
    plt.ylabel('Total Profit')
    plot_filepath_profit_year = os.path.join(app.config['PLOTS_FOLDER'], 'profit_per_year.png')
    plt.savefig(plot_filepath_profit_year)
    plt.close()

    # Plot total sales per year
    total_sales_per_year = sales_data.groupby(sales_data.index.year)['Total Revenue'].sum()
    plt.figure(figsize=(10, 6))
    total_sales_per_year.plot(kind='bar', color='blue')
    plt.title('Total Sales per Year')
    plt.xlabel('Year')
    plt.ylabel('Total Sales')
    plot_filepath_sales_year = os.path.join(app.config['PLOTS_FOLDER'], 'sales_per_year.png')
    plt.savefig(plot_filepath_sales_year)
    plt.close()
    
    # Prepare comments
    comments = {
        'most_profitable_item': f"The item making the most profit is {most_profitable_item} with a total profit of {most_profitable_value}.",
        'least_profitable_item': f"The item with the least profit is {least_profitable_item} with a total profit of {least_profitable_value}.",
        'highest_sales_country': f"The country with the highest sales is {highest_sales_country} with a total revenue of {highest_sales_value}.",
        'max_profit_year': f"The year with the highest profit is {max_profit_year} with a total profit of {max_profit_value}.",
        'advice': "Consider focusing more on the high-profit items and exploring why certain items are underperforming. Understanding the market trends in the highest sales country can help increase overall revenue."
    }
    
    return render_template('results.html', 
                           monthly_sales_plot=url_for('static', filename='plots/monthly_sales.png'), 
                           category_sales_plot=url_for('static', filename='plots/category_sales.png'),
                           prediction_plot=url_for('static', filename='plots/sales_prediction.png'),
                           revenue_by_country_plot=url_for('static', filename='plots/revenue_by_country.png'),
                           units_vs_cost_plot=url_for('static', filename='plots/units_vs_cost.png'),
                           bar_plot=url_for('static', filename='plots/bar_plot.png'),
                           barh_plot=url_for('static', filename='plots/barh_plot.png'),
                           profit_by_item_plot=url_for('static', filename='plots/profit_by_item.png'),
                           profit_per_year_plot=url_for('static', filename='plots/profit_per_year.png'),
                           sales_per_year_plot=url_for('static', filename='plots/sales_per_year.png'),
                           comments=comments)

if __name__ == '__main__':
    app.run(debug=True)
