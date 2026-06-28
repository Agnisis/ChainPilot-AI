import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Bar, Doughnut, Line, Radar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { position: "top" },
  },
  scales: {
    x: { ticks: { maxTicksLimit: 12 } },
  },
};

function toChartJsData(chart) {
  return {
    labels: chart.labels,
    datasets: chart.datasets.map((ds) => ({
      label: ds.label,
      data: ds.data,
      borderColor: ds.borderColor,
      backgroundColor: ds.backgroundColor || ds.borderColor,
      fill: ds.fill ?? false,
      tension: 0.25,
    })),
  };
}

export default function ChartPanel({ chart }) {
  const data = toChartJsData(chart);
  const options = { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: chart.title } } };

  let Component = Line;
  if (chart.type === "bar") Component = Bar;
  if (chart.type === "doughnut") Component = Doughnut;
  if (chart.type === "radar") Component = Radar;

  return (
    <div className="chart-card">
      <div className="chart-container">
        <Component data={data} options={options} />
      </div>
      {chart.interpretation && <p className="chart-note">{chart.interpretation}</p>}
    </div>
  );
}
