"""
Flask Web Application for Customer Segmentation + ML Clustering + Churn Prediction
"""

import os
import pickle
import uuid
import zipfile
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    session,
    has_request_context,
)
from werkzeug.utils import secure_filename

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

from segmentation import full_pipeline

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = os.urandom(32)

UPLOAD_FOLDER = 'uploads'
SESSION_DATA_FOLDER = os.path.join(UPLOAD_FOLDER, 'session_data')
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_DATA_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_session_id():
    sid = session.get('session_id')
    if not sid:
        sid = uuid.uuid4().hex
        session['session_id'] = sid
    return sid


def _session_state_path():
    return os.path.join(SESSION_DATA_FOLDER, f"{_get_session_id()}.pkl")


def _save_state(state):
    with open(_session_state_path(), 'wb') as f:
        pickle.dump(state, f)


def _load_state():
    path = _session_state_path()
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _chart_path(filename):
    return os.path.join(STATIC_FOLDER, filename)


def _chart_url(filename):
    path = _chart_path(filename)
    if not os.path.exists(path):
        return None
    return url_for('static', filename=filename, v=int(os.path.getmtime(path)))


def generate_feature_importance_chart(model):
    if model is None or not hasattr(model, 'coef_'):
        return

    coeffs = np.ravel(model.coef_)
    if coeffs.size < 3:
        return

    labels = ['Recency', 'Frequency', 'Revenue']
    values = [coeffs[0], coeffs[1], coeffs[2]]
    colors = ['#ef4444' if v < 0 else '#0f766e' for v in values]

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(labels, values, color=colors)
    plt.title('Churn Model Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.axhline(0, color='#334155', linewidth=1)
    plt.tight_layout()
    plt.savefig(_chart_path('feature_importance.png'), dpi=140)
    plt.close()


# --------------------------------------------------
# SEGMENTATION PIPELINE
# --------------------------------------------------

def process_segmentation(df, revenue_col):

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", format="mixed", dayfirst=True)
    df = df.dropna(subset=["InvoiceDate"])

    if revenue_col != "Revenue":
        df["Revenue"] = df[revenue_col]

    rfm = full_pipeline(df, static_dir=STATIC_FOLDER)

    return rfm


# --------------------------------------------------
# CHURN MODEL
# --------------------------------------------------

def train_churn_model(rfm_df):

    rfm_df = rfm_df.copy()
    rfm_df['Churn'] = (rfm_df['Recency'] > 180).astype(int)

    feature_cols = ['Recency', 'Frequency', 'Monetary', 'AOV']

    X = rfm_df[feature_cols].fillna(0)
    y = rfm_df['Churn']

    if len(y.unique()) <= 1:
        return np.zeros(len(y)), None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_counts = y.value_counts()

    model = LogisticRegression(max_iter=1000)

    # For small/imbalanced data, train on full set to avoid split failures.
    if len(rfm_df) < 10 or class_counts.min() < 2:
        try:
            model.fit(X_scaled, y)
            probs = model.predict_proba(X_scaled)[:, 1]
            return probs, model, scaler
        except ValueError:
            return np.zeros(len(y)), None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_scaled)[:, 1]

    return probs, model, scaler


def predict_churn(state):

    if 'df' not in state:
        return state

    df = state['df']

    if (
        'Churn_Probability' not in df.columns
        or 'churn_model' not in state
        or 'churn_scaler' not in state
    ):
        probs, model, scaler = train_churn_model(df)

        df['Churn_Probability'] = probs

        def risk(p):
            if p >= 0.7:
                return "High Risk"
            elif p >= 0.4:
                return "Medium Risk"
            elif p >= 0.2:
                return "Low Risk"
            return "Safe"

        df['Risk_Level'] = df['Churn_Probability'].apply(risk)

        state['df'] = df
        state['churn_model'] = model
        state['churn_scaler'] = scaler
        if model is not None:
            generate_feature_importance_chart(model)

        if has_request_context():
            _save_state(state)
    else:
        model = state.get('churn_model')
        if model is not None:
            generate_feature_importance_chart(model)

    return state


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            lower_filename = filename.lower()
            if lower_filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            required = ['CustomerID', 'InvoiceDate', 'InvoiceNo']
            for col in required:
                if col not in df.columns:
                    flash(f'Missing column {col}', 'error')
                    return redirect(url_for('index'))

            revenue_col = None
            for col in ['Revenue', 'Total', 'Amount', 'Price']:
                if col in df.columns:
                    revenue_col = col
                    break

            if revenue_col is None:
                flash("Revenue column not found", 'error')
                return redirect(url_for('index'))

            state = {
                'filename': filename,
                'filepath': filepath,
                'revenue_col': revenue_col,
                'preview_rows': df.head(10).fillna('').to_dict('records'),
                'preview_columns': df.columns.tolist(),
                'row_count': int(len(df)),
                'col_count': int(df.shape[1]),
                'unique_customers': int(df['CustomerID'].nunique()),
                'total_missing_values': int(df.isnull().sum().sum()),
                'missing_values': {k: int(v) for k, v in df.isna().sum().to_dict().items()}
            }

            _save_state(state)

            return redirect(url_for('preview'))

        except Exception as e:
            flash(str(e), 'error')
            return redirect(url_for('index'))

    flash('Invalid file', 'error')
    return redirect(url_for('index'))


@app.route('/results')
def results():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    if 'df' not in state:
        return redirect(url_for('preview'))

    state = predict_churn(state)
    df = state['df']

    segment_counts = df['Segment'].value_counts()
    cluster_counts = df['Cluster'].value_counts().sort_index()

    summary = {
        'total_customers': len(df),
        'avg_recency': round(df['Recency'].mean(), 1),
        'avg_frequency': round(df['Frequency'].mean(), 1),
        'avg_monetary': round(df['Monetary'].mean(), 2),
        'avg_churn_prob': round(df['Churn_Probability'].mean() * 100, 1)
    }

    at_risk_customers = df.nlargest(10, 'Churn_Probability').to_dict('records')
    total_customers = max(1, len(df))
    champions_count = int(segment_counts.get('Champions', 0))
    at_risk_count = int(segment_counts.get('At Risk', 0))
    hibernating_count = int(segment_counts.get('Hibernating', 0))
    champions_percentage = round((champions_count / total_customers) * 100, 1)

    insights = [
        f'Champions customers represent {champions_percentage}% of the customer base.',
        f'At Risk customers identified: {at_risk_count}.',
        f'Hibernating customers identified: {hibernating_count}.'
    ]

    if champions_percentage > 10:
        insights.append('Strong Champions share detected: prioritize VIP loyalty programs and premium retention offers.')
    if at_risk_count > 0:
        insights.append('At Risk segment exists: launch targeted retention campaigns with personalized win-back journeys.')
    if hibernating_count > 0:
        insights.append('Hibernating customers found: run re-engagement campaigns with time-bound incentives.')

    highest_churn_customers = df.nlargest(3, 'Churn_Probability')['CustomerID'].astype(str).tolist()
    if highest_churn_customers:
        insights.append(f'Highest churn-probability customers: {", ".join(highest_churn_customers)}.')

    return render_template(
        'results.html',
        filename=state['filename'],
        segment_counts=segment_counts.to_dict(),
        cluster_counts=cluster_counts.to_dict(),
        summary=summary,
        at_risk_customers=at_risk_customers,
        insights=insights
    )


@app.route('/customers')
def customers():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    if 'df' not in state:
        return redirect(url_for('preview'))

    state = predict_churn(state)
    df = state['df']

    selected_segment = request.args.get('segment', 'All')
    page = request.args.get('page', 1, type=int)
    per_page = 25

    segments = ['All', 'Champions', 'Loyal', 'At Risk', 'Hibernating', 'Others']
    if selected_segment not in segments:
        selected_segment = 'All'

    filtered_df = df
    if selected_segment != 'All':
        filtered_df = df[df['Segment'] == selected_segment]

    total = len(filtered_df)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(max(page, 1), total_pages)

    start = (page - 1) * per_page
    end = start + per_page
    page_df = filtered_df.iloc[start:end]

    return render_template(
        'customers.html',
        customers=page_df.to_dict('records'),
        segments=segments,
        current_segment=selected_segment,
        page=page,
        total_pages=total_pages
    )


@app.route('/segment/<segment_name>')
def segment_detail(segment_name):

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    if 'df' not in state:
        return redirect(url_for('preview'))

    state = predict_churn(state)
    df = state['df']

    seg_df = df[df['Segment'] == segment_name]
    if seg_df.empty:
        flash(f'Segment "{segment_name}" not found.', 'error')
        return redirect(url_for('results'))

    stats = {
        'count': int(len(seg_df)),
        'avg_recency': round(seg_df['Recency'].mean(), 1),
        'avg_frequency': round(seg_df['Frequency'].mean(), 1),
        'avg_monetary': round(seg_df['Monetary'].mean(), 2),
        'total_revenue': round(seg_df['Monetary'].sum(), 2),
        'avg_churn_prob': round(seg_df['Churn_Probability'].mean() * 100, 1),
    }

    top_customers = seg_df.nlargest(20, 'Monetary').to_dict('records')

    return render_template(
        'segment_detail.html',
        segment_name=segment_name,
        stats=stats,
        top_customers=top_customers
    )


@app.route('/charts')
def charts():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    if 'df' not in state:
        return redirect(url_for('preview'))

    state = predict_churn(state)
    df = state['df']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    seg = df['Segment'].value_counts()
    axes[0].pie(seg.values, labels=seg.index, autopct='%1.1f%%')
    axes[0].set_title("Customer Segment Distribution")

    clusters = df['Cluster'].value_counts().sort_index()
    axes[1].bar(clusters.index.astype(str), clusters.values)
    axes[1].set_title("ML Cluster Distribution")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Customers")

    plt.tight_layout()

    overview_path = _chart_path('overview_charts.png')
    plt.savefig(overview_path, format='png', dpi=140)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.pie(seg.values, labels=seg.index, autopct='%1.1f%%')
    plt.title("Customer Segment Distribution")
    plt.tight_layout()
    plt.savefig(_chart_path('segment_pie.png'), dpi=140)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(clusters.index.astype(str), clusters.values, color='#0f766e')
    plt.title("ML Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(_chart_path('cluster_distribution.png'), dpi=140)
    plt.close()

    return render_template(
        "charts.html",
        overview_chart_url=_chart_url('overview_charts.png'),
        cluster_scatter_url=_chart_url('customer_clusters.png'),
        feature_importance_url=_chart_url('feature_importance.png')
    )


@app.route('/download-charts')
def download_charts():
    chart_files = [
        'segment_pie.png',
        'cluster_distribution.png',
        'customer_clusters.png',
        'feature_importance.png'
    ]
    existing = [name for name in chart_files if os.path.exists(_chart_path(name))]

    if not existing:
        flash('No charts available to download yet.', 'error')
        return redirect(url_for('charts'))

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name in existing:
            zf.write(_chart_path(name), arcname=name)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name='customer_analytics_charts.zip',
        mimetype='application/zip'
    )


@app.route('/download')
def download():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    if 'df' not in state:
        return redirect(url_for('preview'))

    state = predict_churn(state)
    df = state['df']

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='SegmentationResults')
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name='segmentation_results.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/preview')
def preview():
    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    if 'df' in state:
        return redirect(url_for('results'))

    return render_template(
        'preview.html',
        filename=state.get('filename'),
        preview_rows=state.get('preview_rows', []),
        preview_columns=state.get('preview_columns', []),
        row_count=state.get('row_count', 0),
        col_count=state.get('col_count', 0),
        unique_customers=state.get('unique_customers', 0),
        total_missing_values=state.get('total_missing_values', 0),
        missing_values=state.get('missing_values', {})
    )


@app.route('/run-segmentation', methods=['POST'])
def run_segmentation():
    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    filepath = state.get('filepath')
    revenue_col = state.get('revenue_col')
    filename = state.get('filename')

    if not filepath or not revenue_col or not os.path.exists(filepath):
        flash('Uploaded file is missing. Please upload again.', 'error')
        return redirect(url_for('index'))

    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        result_df = process_segmentation(df, revenue_col)
        state['df'] = result_df
        _save_state(state)

        return redirect(url_for('results'))
    except Exception as e:
        flash(str(e), 'error')
        return redirect(url_for('preview'))


@app.route('/reset')
def reset():
    path = _session_state_path()
    if os.path.exists(path):
        os.remove(path)
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
