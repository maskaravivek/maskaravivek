var height = 600;
var width = 700;
var margin = ({ top: 20, right: 60, bottom: 40, left: 80 });
var boundedWidth = width - margin.left - margin.right;
var boundedHeight = height - margin.top - margin.bottom;

var trump_all;
var trump_republicans;
var obama_all;
var obama_democrats;
var data;

var f = d3.format(".0%");

var colors = d3.scaleOrdinal()
  .range(["#81b29a", "#e07a5f", "#e5e5e5"])

// This runs when the page is loaded
document.addEventListener('DOMContentLoaded', function () {

  d3.select("#candidate").on("input", function () {
    reloadCharts()
  });

  // Load both files before doing anything else
  Promise.all([d3.csv('data/processed/trump_all.csv'),
  d3.csv('data/processed/trump_republicans.csv'),
  d3.csv('data/processed/obama_all.csv'),
  d3.csv('data/processed/obama_democrats.csv')])
    .then(function (values) {

      trump_all = values[0];
      trump_republicans = values[1];
      obama_all = values[2];
      obama_democrats = values[3];

      data = values[2];

      drawBarChart('#leftchart', trump_republicans, "(*Avg. Amongst Republicans)");
      drawBarChart('#rightchart', trump_all, "(*Overall Avg.)");
    })
});

function reloadCharts() {
  var candidate = document.getElementById("candidate").value;

  document.getElementById('leftchart').innerHTML = "<div></div>"
  document.getElementById('rightchart').innerHTML = "<div></div>"

  if (candidate == 'obama') {
    drawBarChart('#leftchart', obama_democrats, "(*Avg. Amongst Democrats)");
    drawBarChart('#rightchart', obama_all, "(*Overall Avg.)");
    document.getElementById('page-title').innerText = "#6: Obama's Approval Ratings"
  } else {
    drawBarChart('#leftchart', trump_republicans, "(*Avg. Amongst Republicans)");
    drawBarChart('#rightchart', trump_all, "(*Overall Avg.)");
    document.getElementById('page-title').innerText = "#6: Trump's Approval Ratings"
  }
}

// Draw the map in the #map svg
function drawBarChart(domElementId, dataset, title) {
  var candidate = document.getElementById("candidate").value;

  if(candidate == 'obama') {
    document.getElementById('left-caption').innerText = 'Approval Ratings amongst Democrats'
  } else {
    document.getElementById('left-caption').innerText = 'Approval Ratings amongst Republicans'
  }
  var keys = dataset.columns.slice(1)
  var metricAccessor = d => d.survey_organization;
  var groupedValues = d => keys.map(key => ({ key: key, value: d[key] }))

  var approval = 0
  for (var index = 0; index < dataset.length; index++) {
    approval = approval + parseFloat(dataset[index]['mean_approval_percent'])
  }
  approval = approval / dataset.length;
  approval = (Math.round(approval * 100) / 100).toFixed(0) + "%";
  var yAccessor = d => d3.max(keys, key => +d[key])

  var yScale = d3.scaleLinear()
    .domain([0, 100])
    .range([boundedHeight, 0])
    .nice()

  var xScale0 = d3.scaleBand()
    .domain(dataset.map(metricAccessor))
    .range([0, boundedWidth])
    .padding(0.2)

  var xScale1 = d3.scaleBand()
    .domain(keys)
    .range([0, xScale0.bandwidth()])
    .padding(0.1);

  var chart = d3.select(domElementId)
    .append("svg")
    .attr("width", width)
    .attr("height", height)

  chart.selectAll("*").remove()

  var bounds = chart.append("g")
    .style("transform", `translate(${margin.left}px, ${margin.top}px)`)

  var yAxisGenerator = d3.axisLeft()
    .scale(yScale)
    .tickFormat(d3.format(".2s"));

  var xAxisGenerator = d3.axisBottom()
    .scale(xScale0);

  var xAxis = bounds.append("g")
    .call(xAxisGenerator)
    .style("transform", `translateY(${boundedHeight}px)`);

  var yAxis = bounds.append("g")
    .call(yAxisGenerator);

  var yAxisLabel = yAxis.append("text")
    .attr("x", -boundedHeight / 2)
    .attr("y", -margin.left + 10)
    .attr("transform", "rotate(-90)")
    .attr("text-anchor", "middle")
    .attr("class", "axisLabel")
    .attr("font-size", "1.5em")
    .attr("fill", "black")
    .text("Percentages");

  var xAxisLabel = xAxis.append("text")
    .attr("x", boundedWidth / 2)
    .attr("y", margin.bottom - 10)
    .attr("text-anchor", "middle")
    .attr("class", "axisLabel")
    .attr("font-size", "1.5em")
    .attr("fill", "black")
    .text("Survey Organization");

  var color = "#DE0100";

  console.log(candidate)
  if (candidate === 'obama') {
    color = "#1405BD";
  }

  chart.append("text")
    .attr("class", "year label")
    .attr("text-anchor", "end")
    .attr("y", 60)
    .attr("x", (width / 2) + 60)
    .attr("fill", color)
    .text(approval);

  chart.append("text")
    .attr("y", 60)
    .attr("x", width / 2 + 120)
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .style("font-size", "10px")
    .attr("font-family", "sans-serif")
    .attr("font-weight", 300)
    .text(title);

  var legend = chart.selectAll(".legend")
    .data(keys)
    .enter().append("g")
    .attr("class", "legend")
    .attr("transform", (d, i) => `translate(0,` + i * 20 + `)`)
    .style("opacity", 0)

  legend.append("rect")
    .attr("x", width - 16)
    .attr("width", 16)
    .attr("height", 16)
    .style("fill", d => colors(d))

  legend.append("text")
    .attr("x", width - 20)
    .attr("y", 9)
    .attr("dy", ".35em")
    .attr("font-size", "0.875em")
    .attr("font-family", "sans-serif")
    .style("text-anchor", "end")
    .text(d => {
      if (d == 'mean_approval_percent') {
        return "Approval"
      } else if (d == "mean_disapprove_percent") {
        return "Disapproval"
      } else {
        return "Undecided"
      }
    })

  legend.transition()
    .delay((d, i) => 1000 + 100 * i)
    .duration(500)
    .style("opacity", 1)

  var slice = bounds.selectAll(".slice")
    .data(dataset)
    .enter().append("g")
    .attr("class", "g")
    .attr("transform", d => `translate(` + xScale0(d.survey_organization) + `,0)`)

  slice.selectAll("rect")
    .data(groupedValues)
    .join("rect")
    .attr("width", xScale1.bandwidth())
    .attr("x", d => xScale1(d.key))
    .attr("fill", d => colors(d.key))
    .attr("height", d => boundedHeight - yScale(0))
    .attr("y", d => yScale(0))

  slice.selectAll("rect")
    .transition()
    .delay(d => Math.random() * 1000)
    .duration(1000)
    .attr("y", d => yScale(d.value))
    .attr("height", d => boundedHeight - yScale(d.value))

  slice.selectAll("text")
    .data(groupedValues)
    .enter().append("text")
    .attr("fill", "#d0743c")
    .attr("x", function (d) { return xScale1(d.key); })
    .attr("y", function (d) { return yScale(+d.value + 2); })
    .text(function (d) { return f(+d.value / 100.0) })
}