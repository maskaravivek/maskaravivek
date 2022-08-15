
var mapSvg;
var mapWidth;
var mapHeight;
var barHeight = 20;
var mapMargin = { top: 20, right: 60, bottom: 60, left: 100 };

var lineSvg;
var lineWidth;
var lineHeight;
var lineInnerHeight;
var lineInnerWidth;
var lineMargin = { top: 20, right: 60, bottom: 60, left: 100 };

var mapData;
var timeData;

var tooltip = d3.select("body")
  .append("div")
  .attr('class', 'tooltip')
  .style("opacity", 1);

// This runs when the page is loaded
document.addEventListener('DOMContentLoaded', function () {
  mapSvg = d3.select('#map');
  lineSvg = d3.select('#linechart');
  mapWidth = +mapSvg.style('width').replace('px', '');
  mapHeight = +mapSvg.style('height').replace('px', '');;
  lineWidth = +lineSvg.style('width').replace('px', '');
  lineHeight = +lineSvg.style('height').replace('px', '');;
  lineInnerWidth = lineWidth - lineMargin.left - lineMargin.right;
  lineInnerHeight = lineHeight - lineMargin.top - lineMargin.bottom;

  // Load both files before doing anything else
  Promise.all([d3.json('data/africa.geojson'),
  d3.csv('data/africa_gdp_per_capita.csv')])
    .then(function (values) {

      mapData = values[0];
      timeData = values[1];

      mapSvg.selectAll("*").remove()
      lineSvg.selectAll("*").remove()
      drawMap();
    })

  d3.select("#year-input").on("input", function () {
    mapSvg.selectAll("*").remove()
    lineSvg.selectAll("*").remove()
    drawMap()
  });
  d3.select("#color-scale-select").on("input", function () {
    mapSvg.selectAll("*").remove()
    lineSvg.selectAll("*").remove()
    drawMap()
  });

});

// Get the min/max values for a year and return as an array
// of size=2. You shouldn't need to update this function.
function getExtentsForYear(yearData) {
  var max = Number.MIN_VALUE;
  var min = Number.MAX_VALUE;
  for (var key in yearData) {
    if (key == 'Year')
      continue;
    let val = +yearData[key];
    if (val > max)
      max = val;
    if (val < min)
      min = val;
  }
  return [min, max];
}

// Draw the map in the #map svg
function drawMap() {
  // create the map projection and geoPath
  let projection = d3.geoMercator()
    .scale(400)
    .center(d3.geoCentroid(mapData))
    .translate([+mapSvg.style('width').replace('px', '') / 2,
    +mapSvg.style('height').replace('px', '') / 2.3]);
  let path = d3.geoPath()
    .projection(projection);

  // get the selected year based on the input box's value
  var year = document.getElementById("year-input").value;

  // get the GDP values for countries for the selected year
  let yearData = timeData.filter(d => d.Year == year)[0];

  // get the min/max GDP values for the selected year
  let extent = getExtentsForYear(yearData);

  // get the selected color scale based on the dropdown value
  var colorScale = d3.scaleSequential(getColorScale())
    .domain(extent);


  // draw the map on the #map svg
  let g = mapSvg.append('g');
  g.selectAll('path')
    .data(mapData.features)
    .enter()
    .append('path')
    .attr('d', path)
    .attr('id', d => { return d.properties.name })
    .attr('class', 'countrymap')
    .style('fill', d => {
      let val = +yearData[d.properties.name];
      if (isNaN(val))
        return 'white';
      return colorScale(val);
    })
    .on('mouseover', function (d, i) {
      d3.select(this)
        .style('stroke', 'cyan')
        .style('stroke-width', '4');

      tooltip.style("opacity", 1);
      let toolTipData = 'Country: ' + d.properties.name + "<br/>" + " GDP: " + getGDP(d.properties.name);
      tooltip.html(toolTipData)
        .style("left", (d3.event.pageX + 10) + "px")
        .style("top", (d3.event.pageY - 15) + "px");
    })
    .on('mousemove', function (d, i) {

    })
    .on('mouseout', function (d, i) {
      d3.select(this)
        .style('stroke', 'black')
        .style('stroke-width', '1');
      tooltip.style("opacity", 0);
    })
    .on('click', function (d, i) {
      drawLineChart(d.properties.name)
    });

  addMapLegend()
}

function getGDP(country) {
  var year = document.getElementById("year-input").value;
  filtered = timeData.filter(function (d) { return (d.Year == year); });
  return +filtered[0][country];
}

function getMaxGDP() {
  var year = document.getElementById("year-input").value;
  filtered = timeData.filter(function (d) { return (d.Year == year); })[0];

  var max = 0;
  Object.keys(filtered).forEach(function (key) {
    max = Math.max(max, filtered[key])
  });

  return max;
}

function getGDPByCountry(country) {
  return timeData.map(function (item) {
    return {
      Year: item.Year,
      Gdp: item[country]
    }
  });
}


function addMapLegend() {
  var colorScale = getColorScale();
  colorScale = d3.scaleSequential([0, getMaxGDP()], colorScale)
  axisScale = d3.scaleLinear()
    .domain([0, getMaxGDP()])
    .range([mapMargin.left, 360 - mapMargin.right])

  axisBottom = g => g
    .attr("class", 'x-axis')
    .attr("transform", "translate(-40," + (520 + barHeight) + ")")
    .style("color", "white")
    .call(d3.axisBottom(axisScale)
      .ticks(5)
      .tickSize(-barHeight))
    .selectAll(".x-axis text")
    .attr("fill", "black")

  mapSvg.select("defs").remove();
  const defs = mapSvg.append("defs");

  const linearGradient = defs.append("linearGradient")
    .attr("id", "linear-gradient");

  linearGradient.selectAll("stop")
    .data(colorScale.ticks().map((t, i, n) => ({ offset: `${100 * i / n.length}%`, color: colorScale(t) })))
    .enter().append("stop")
    .attr("offset", d => d.offset)
    .attr("stop-color", d => d.color);

  mapSvg.append('g')
    .attr("transform", `translate(-40, 520)`)
    .append("rect")
    .attr('transform', `translate(${mapMargin.left}, 0)`)
    .attr("width", 360 - mapMargin.right - mapMargin.left)
    .attr("height", barHeight)
    .style("fill", "url(#linear-gradient)");

  mapSvg.append('g')
    .call(axisBottom);
}

function getColorScale() {
  var scale = document.getElementById("color-scale-select").value;
  if (scale === "interpolateRdYlGn") {
    return d3.interpolateRdYlGn;
  } else if (scale === "interpolateViridis") {
    return d3.interpolateViridis;
  } else if (scale === "interpolateBrBG") {
    return d3.interpolateBrBG;
  } else if (scale === "interpolateRdGy") {
    return d3.interpolateRdGy;
  } else {
    return d3.interpolateRdYlBu;
  }
}

// Draw the line chart in the #linechart svg for
// the country argument (e.g., `Algeria').
function drawLineChart(country) {

  if (!country)
    return;
  lineSvg.selectAll("*").remove()

  lineSvg
    .attr("width", lineWidth)
    .attr("height", lineHeight)
    .append("g")
    .attr("transform",
      "translate(" + lineMargin.left + "," + lineMargin.top + ")");

  var g = lineSvg
    .attr("width", lineInnerWidth)
    .attr("height", lineInnerHeight)
    .append("g")
    .attr("class", "lineAxis")
    .attr("transform", `translate(${lineMargin.left}, ${lineMargin.top})`)

  var gdpData = getGDPByCountry(country)

  var x = d3.scaleLinear()
    .range([0, lineInnerWidth]);

  x.domain(d3.extent(timeData, function (d) {
    return +d.Year;
  }));

  var xAxis = g.append("g")
    .attr("transform", "translate(0," + lineInnerHeight + ")")
    .attr("class", "lineAxis")
    .call(d3.axisBottom(x).ticks(10));

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([0, d3.max(gdpData, function (d) { return +d.Gdp; })])
    .range([lineInnerHeight, 0]);

  g.append("g")
    .call(d3.axisRight(y)
      .tickSize(lineInnerWidth))
    .call(g => g.select(".domain")
      .remove()).call(g => g.selectAll(".tick:not(:first-of-type) line")
        .attr("stroke-opacity", 0.5)
        .attr("stroke-dasharray", "5, 10")).call(g => g.selectAll(".tick text")
          .attr("x", 4)
          .attr("dy", -4));

  g.append("text")
    .attr("transform",
      "translate(" + (lineInnerWidth / 2) + " ," +
      (lineInnerHeight + lineMargin.top + 20) + ")")
    .style("text-anchor", "middle")
    .style("font-size", "14px")
    .attr("font-family", "sans-serif")
    .attr("font-weight", 400)
    .text("Year");

  xAxis.selectAll(".tick text")
    .each(function (_, i) {
      if (i % 2 !== 0) d3.select(this).remove();
    });

  g.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", -60)
    .attr("x", 0 - (lineHeight / 2))
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .style("font-size", "14px")
    .attr("font-family", "sans-serif")
    .attr("font-weight", 400)
    .text("GDP for " + country + " (based on current USD)");

  var bisect = d3.bisector(function (d) { return d.Year; }).left;

  var focus = g
    .append('g')
    .append('circle')
    .style("fill", "none")
    .attr("stroke", "black")
    .attr('r', 10)
    .style("opacity", 0)

  // Add the line
  g.append("path")
    .datum(gdpData)
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("stroke-width", 2)
    .attr("d", d3.line()
      .x(function (d) { return x(d.Year) })
      .y(function (d) { return y(d.Gdp) })
    )

  lineSvg.on('mouseover', function (d, i) {
    var x0 = x.invert(d3.mouse(this)[0]);
    var i = bisect(gdpData, x0, 1);
    selectedData = gdpData[i]
    if (selectedData === undefined) {
      return
    }
    focus.style("opacity", 1);
    tooltip.style("opacity", 1);
    let toolTipData = 'Year: ' + selectedData.Year + "<br/> GDP: " + (+selectedData.Gdp);
    tooltip.html(toolTipData)

  }).on('mousemove', function (d, i) {
    var x0 = x.invert(d3.mouse(this)[0]);
    var i = bisect(gdpData, x0, 1);
    selectedData = gdpData[i]

    if (selectedData === undefined) {
      return
    }

    focus
      .attr("cx", x(selectedData.Year))
      .attr("cy", y(selectedData.Gdp))
    let toolTipData = 'Year: ' + selectedData.Year + "<br/> GDP: " + (+selectedData.Gdp);
    tooltip.html(toolTipData)
      .style("left", (d3.event.pageX + 10) + "px")
      .style("top", (d3.event.pageY - 15) + "px");
  })
    .on('mouseout', function (d, i) {
      focus.style("opacity", 0)
      tooltip.style("opacity", 0);
    })
}
