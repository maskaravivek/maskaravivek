/* innovative_viz.js */
import { drawHeatMap } from './heatMap.js';
import { drawLineChart } from './lineChart.js'

var scatterPlotMargin = { top: 10, right: 40, bottom: 40, left: 60 }

var scatterPlotWidth = 1000 - scatterPlotMargin.left - scatterPlotMargin.right;
var scatterPlotHeight = 500 - scatterPlotMargin.top - scatterPlotMargin.bottom;

var symbol = d3.symbol().size(150);

var scatterPlotSvg = d3.select("#innovative_dataviz")
    .append("svg")
    .attr("id", "inno_viz")
    .attr("width", scatterPlotWidth + scatterPlotMargin.left + scatterPlotMargin.right)
    .attr("height", scatterPlotHeight + scatterPlotMargin.top + scatterPlotMargin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + scatterPlotMargin.left + "," + scatterPlotMargin.top + ")");

d3.select(".playPause").on("click", function () {
    toggleAnimation()
});

var compass = d3.select("#innovative_dataviz").select("svg").append("g")
    .attr("transform", "translate(" + 0 + "," + (300) + ")")

var tooltip = d3.select("body")
    .append("div")
    .attr('class', 'tooltip')
    .style("opacity", 0);

var factory;

// data
var data;
var windData;
var completeData;
var smokeLocations;
var factoryEmissions;

var levels = 0;
// svg elements
var gDots;
var ind = 0;
var curDateBg;
// axes
var x = d3.scaleLinear()
    .range([0, scatterPlotWidth]);

var y = d3.scaleLinear()
    .range([scatterPlotHeight, 0]);

const t = d3.transition().duration(100)

const colorScale = d3.scaleOrdinal()
    .range(d3.schemeCategory10);

const colorScaleEmissions = d3.scaleSequential()
    .interpolator(d3.interpolateGreys);

var running = false;
var timer;

const day = 86400000;
const now = new Date(2016, 0);
const year = now.getFullYear();
let calender_data = [];

var windG = null

var windLines = [];

document.addEventListener('DOMContentLoaded', function () {
    drawInitialViz()
});

function drawInitialViz() {
    loadData()
}

function loadData() {
    Promise.all([d3.csv('data/sensor_locations.csv'),
    d3.csv('data/windData.csv'),
    d3.csv('data/complete_data.csv'), d3.csv('data/smoke_locations.csv'),
    d3.csv('data/factoryEmissions.csv')]).then(function (values) {
        data = values[0];
        windData = values[1];
        completeData = values[2];
        smokeLocations = values[3];
        factoryEmissions = values[4];

        x.domain([0, d3.max(data, function (d) {
            return +d["x"];
        })]);

        y.domain([0, d3.max(data, function (d) {
            return +d["y"];
        })]);

        colorScaleEmissions.domain([-50, 151.8132489])

        prepareWindData()

        // drawCircles()
        drawScatterPlot()
        drawCompass([windData[ind]])
    });

    // On chemical change call heat map for new chemical
    d3.select('#chemical').on('change', function () {
        drawHeatMap(completeData, factory);
    });
    d3.select("#months-innovative_dataviz").on('change', function () {
        drawScatterPlot()
    });
}

function prepareWindData() {
    var windSpeed = windData[ind].speed

    var xCoord = [40, 50, 60, 70, 80, 90, 100]
    var yCoord = [0, 10, 20, 30]
    var windDelay = [100, 200, 300, 400]

    windLines = []
    // y = x

    for (var i = 0; i < xCoord.length; i++) {
        for (var j = 0; j < yCoord.length; j++) {
            var line = {
                x0: xCoord[i],
                y0: yCoord[j],
                x1: xCoord[i],
                y1: yCoord[j] + 2,
                s: windSpeed,
                duration: 200 / windSpeed, /* pre-compute duration */
                delay: windDelay[j] /* pre-compute delay */
            }
            windLines.push(line)
        }
    }
}

function lineAnimate(selection) {
    selection
        .attr('x2', function (d) { return x(d.x1) })
        .attr('y2', function (d) { return y(d.y1) })
        .style('opacity', 0)
        .transition()
        .ease(d3.easeLinear)
        .duration(function (d) { return d.duration; })
        .delay(function (d) { return d.delay; })
        .attr('x2', function (d) { return x(d.x1) })
        .attr('y2', function (d) { return y(d.y1) })
        .style('opacity', 0.4)
        .transition()
        .duration(800)
        .style('opacity', 0.1)
        .on('end', function () { d3.select(this).call(lineAnimate) });
}

function drawWind() {
    scatterPlotSvg.selectAll('line').remove()
    windG = scatterPlotSvg.append('g')
    windG.selectAll('line')
        .data(windLines)
        .enter()
        .append("line")
        .attr('x1', function (d) { return x(d.x0) })
        .attr('y1', function (d) { return y(d.y0) })
        .attr('class', 'gaugeChart-needle')
        .attr('stroke-width', 0.5)
        .attr("stroke", "#D3D3D3")
        .attr("marker-end", "url(#triangle)")
        .call(lineAnimate)
        .attr("transform", d => 'rotate(' + windData[ind]['direction'] + ' ' + (scatterPlotWidth / 2 + 50) + ' ' + (scatterPlotHeight / 2 + 40) + ')');
}

function drawScatterPlot() {
    scatterPlotSvg.selectAll("*").remove()

    drawWind();

    scatterPlotSvg.append("text")
        .attr("x", (scatterPlotWidth / 2))
        .attr("y", 10 - (scatterPlotMargin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .attr("font-family", "sans-serif")
        .style("text-decoration", "underline")
        .attr("font-weight", 700)
        .text("Sensor & Factory Locations");

    drawSmoke()

    var sensorFactories = scatterPlotSvg.append("g")
        .attr("stroke-width", 1.5)
        .attr("font-family", "sans-serif")
        .attr("font-size", 10)
        .selectAll("path")
        .data(data)
        .join("image")
        .attr("transform", d => `translate(${x(d.x - 30)},${y(d.y - 2)})`)
        .attr("fill", d => colorScale(d.type))
        .attr("xlink:href", d => {
            if (d.type == 'sensor') {
                return 'images/warning.png'
            } else {
                return 'images/factory.png'
            }
        })

    sensorFactories.on('mouseover', function (d, i) {
        var displayText = d.name;
        if (d.type == 'sensor') {
            displayText = "Sensor " + displayText
        }

        tooltip.style("opacity", 1);
        tooltip.html(displayText)
            .style("left", (d3.event.pageX + 10) + "px")
            .style("top", (d3.event.pageY - 15) + "px");
    })
        .on('mouseout', function (d, i) {
            tooltip.style("opacity", 0);
        })
        .on('click', function (d, i) {
            if (d.type == 'sensor') {
                d3.select('#factoryPlots').style('display', 'none');
                d3.select('#sensorPlots').style('display', '');
                var month = document.getElementById("months-linechart").value;
                drawLineChart(month, d.name)
            }
            else {
                d3.select('#sensorPlots').style('display', 'none');
                d3.select('#factoryPlots').style('display', '');
                drawHeatMap(completeData, d.name);
                factory = d.name;
            }

            window.scrollTo({ top: 2000, behavior: 'smooth' });
        })

    makeMonth(+document.getElementById("months-innovative_dataviz").value, year);
    drawCalender()

    curDateBg = scatterPlotSvg
        .append("text")
        .attr("class", "curDateBg")
        .attr("x", width / 2 - 200)
        .attr("y", height / 2 + 70)
        .attr("font-size", "16px")
        .attr("text-anchor", "middle")
        .attr("opacity", 0.5)
        .style("pointer-events", "none")
        .text(windData[ind]['Date Time '])

    d3.select("#dateSlider").on("change", function (d) {
        ind = +this.value
        d3.select("#yearEntry").attr("value", ind)
        d3.select("#yearEntry").property("value", ind)
        curDateBg.text(windData[ind]['Date Time '])


        scatterPlotSvg
            .selectAll(".smoke")
            .transition()
            .attr('r', '0px')
            .remove()

        drawSmoke()

        levels += 1
        if (levels == 3) {
            levels = 0
        }

        drawWind()
        drawCompass([windData[ind]])
    })

}

function drawCompass(compassData) {

    compass.selectAll('g')
        .data(compassData)
        .join(
            enter => enterCompass(enter, t),
            update => updateCompass(update, t)
        )
}

function updateCompass(update, t) {

    update.select('#speed')
        .attr("text-anchor", "middle")
        .text(d => "Wind Speed:" + d.speed)

    update.select('#direction')
        .attr("text-anchor", "middle")
        .text(d => "Wind direction:" + d.direction)

    update.select("line")
        .transition()
        .duration(500)
        .attr('transform', d => 'rotate(' + d.direction + ' 100 100)')
        .end()
        .then(circleTransitions);
}

function enterCompass(enter, t) {
    var glyph = enter.append('g')

    glyph.append("circle")
        .attr("transform", "translate(100,100)")
        .attr("r", 50)
        .attr("stroke", "black")
        .attr("fill", "white")
        .style("fill-opacity", 0)

    glyph.append("text")
        .attr("transform", "translate(100,90)")
        .attr("id", "speed")
        .style("text-anchor", "middle")
        .style("font-size", "8px")
        .style("font-weight", "700")
        .style("font-family", "sans-serif")
        .style("fill", "gray")
        .style("opacity", 0.5)
        .text(d => "Wind Speed:" + d.speed)

    glyph.append("text")
        .attr("transform", "translate(100,110)")
        .attr("id", "direction")
        .style("text-anchor", "middle")
        .style("font-size", "8px")
        .style("font-weight", "700")
        .style("font-family", "sans-serif")
        .style("fill", "gray")
        .style("opacity", 0.5)
        .text(d => "Wind direction:" + d.direction)

    glyph.append("svg:defs").append("svg:marker")
        .attr("id", "triangle")
        .attr("refX", 6)
        .attr("refY", 6)
        .attr("markerWidth", 30)
        .attr("markerHeight", 30)
        .attr("markerUnits", "userSpaceOnUse")
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M 0 0 20 6 0 12 3 6")
        .style("fill", "black")

    glyph.append('line')
        .attr("x1", 100)
        .attr("y1", 145)
        .attr("x2", 100)
        .attr("y2", 70)
        .attr('stroke-width', 1)
        .attr("stroke", "black")
        .attr("marker-end", "url(#triangle)")
        .attr('transform', d => 'rotate(' + d.direction + ' 100 100)')
}

function stopAnimation() {
    d3.select('.playPause').attr('value', "Start");
}

function toggleAnimation() {
    const currentState = d3.select('.playPause').attr('value');
    const updatedLabel = currentState == 'Start' ? 'Stop' : 'Start';
    d3.select('.playPause').attr('value', updatedLabel);

    document.getElementById('playPause').innerHTML = updatedLabel
    circleTransitions()
}

function circleTransitions() {
    if (d3.select('.playPause').attr('value') == 'Stop' && ind < windData.length - 1) {
        ind = ind + 1;

        curDateBg.text(windData[ind]['Date Time '])
        d3.select("#dateSlider").attr("value", ind)
        d3.select("#dateSlider").property("value", ind)

        drawWind()
        drawCompass([windData[ind]]);

        scatterPlotSvg
            .selectAll(".smoke")
            .transition()
            .attr('r', '0px')
            .remove()

        drawSmoke()

        levels += 1
        if (levels == 3) {
            levels = 0
        }


    }
    else if (ind == windData.length - 1) {
        ind = 0;
        stopAnimation();
    }
    else stopAnimation();
}

function makeMonth(month, year) {
    const monthDays = [];
    let loopMonth = month;
    let loopDay = 0;
    let loopDate = new Date(year, loopMonth, loopDay);
    let loopStartDay = loopDate.getDay();
    while (loopMonth === month) {
        monthDays.push({ date: loopDate, col: loopDate.getDay(), row: Math.floor((loopDate.getDate() + loopStartDay) / 7) });

        loopDate = new Date(loopDate.getTime() + day);
        loopMonth = loopDate.getMonth();
    }

    if (monthDays[0].date.getDate() > 1) {
        monthDays.shift();
    }
    if (monthDays[0].row > 0) {
        monthDays.forEach(d => {
            --d.row;
            return d;
        });
    }

    calender_data = { month, days: monthDays };
}

function drawCalender() {
    const g = scatterPlotSvg.append("g");

    const outlines = g.append("polygon")
        .datum(calender_data.days)
        .attr("class", "outline calender")
        .attr("fill-opacity", 0);

    const columns = d3.scaleBand()
        .domain(d3.range(0, 7));

    const rows = d3.scaleBand()
        .domain(d3.range(0, 5));

    const days = g.selectAll(".day")
        .data(calender_data.days)
        .enter().append("g")
        .attr("class", "day calender");

    const dayRects = days.append("rect")
        .attr("class", "rect calender")
        .attr("fill-opacity", 0)
        .attr("stroke", "black");

    const dayNums = days.append("text")
        .attr("class", "num calender")
        .text(d => d.date.getDate())
        .attr("dy", 4.5)
        .attr("dx", -8);

    const dayOfWeek = g.selectAll(".day-of-week")
        .data(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
        .enter().append("text")
        .attr("class", "day-of-week calender")
        .attr("font-size", "13px")
        .attr("dy", -4)
        .attr("dx", -10)
        .text(d => d);

    const width = 200
    const height = 200


    g.attr("transform", "translate(-50,50)");

    columns.range([0, width]);

    rows.range([0, height]);

    calender_data.days.forEach(d => {
        d.x0 = columns(d.col);
        d.x1 = d.x0 + columns.bandwidth();
        d.y0 = rows(d.row);
        d.y1 = d.y0 + rows.bandwidth();
        d.v0 = [d.x0, d.y0];

        return d;
    });


    dayOfWeek
        .attr("x", (d, i) => columns(i) + columns.bandwidth() / 2);

    days
        .attr("transform", d => `translate(${d.v0})`);

    dayRects
        .attr("width", columns.bandwidth())
        .attr("height", rows.bandwidth())

    dayRects
        .attr("width", columns.bandwidth())
        .attr("height", rows.bandwidth())

    days
        .on('mouseover', function (d, i) {
            if ([1, 2, 3, 4].includes(+d3.select(this).select("text").text()) && +document.getElementById("months-innovative_dataviz").value == 7) {
            }
            else {
                d3.select(this).select("text")
                    .style("stroke-width", 1)
                    .attr("stroke", "red")
            }
        })
        .on('mouseout', function (d, i) {
            if ([1, 2, 3, 4].includes(+d3.select(this).select("text").text()) && +document.getElementById("months-innovative_dataviz").value == 7) {
            }
            else {
                d3.select(this).select("text")
                    .style("stroke-width", 0)
                    .attr("stroke", "black")
            }
        })
        .on('click', function (d, i) {
            if ([1, 2, 3, 4].includes(+d3.select(this).select("text").text()) && +document.getElementById("months-innovative_dataviz").value == 7) {

            }
            else {
                var dateFromCalender = new Date(2016, +document.getElementById("months-innovative_dataviz").value, +d3.select(this).select("text").text(), 1);
                var temp = []
                windData.forEach(function (d, i) {
                    if (new Date(d["Date Time "]).getTime() === dateFromCalender.getTime()) {
                        temp.push(i);
                    }
                })
                stopAnimation()
                ind = temp[0];
                toggleAnimation()
            }
        })

    dayNums
        .attr("x", columns.bandwidth() / 2)
        .attr("y", rows.bandwidth() / 2);

    outlines
        .attr("points", calcHull);
}

function calcHull(days) {
    const x0min = d3.min(days, d => d.x0),
        x1max = d3.max(days, d => d.x1),
        y0min = d3.min(days, d => d.y0),
        y1max = d3.max(days, d => d.y1);

    const r0 = days.filter(f => f.row === 0),
        r0x0min = d3.min(r0, d => d.x0),
        r0x1max = d3.max(r0, d => d.x1);

    const r4 = days.filter(f => f.row === 4),
        r4x1max = d3.max(r4, d => d.x1),
        r4x0min = d3.min(r4, d => d.x0);

    let points = [[r0x0min, y0min], [r0x1max, y0min]];

    if (r4x1max < x1max) {
        const r3y1 = days.filter(f => f.row === 3)[0].y1;
        points.push([x1max, r3y1]);
        points.push([r4x1max, r3y1]);
    }
    points.push([r4x1max, y1max]);

    points.push([r4x0min, y1max]);

    if (r0x0min > x0min) {
        const r1y0 = days.filter(f => f.row === 1)[0].y0;
        points.push([x0min, r1y0]);
        points.push([r0x0min, r1y0]);
    }

    return points;
}

function drawSmoke() {

    scatterPlotSvg.append("g")
        .selectAll("path")
        .data(smokeLocations)
        .enter()
        .append('circle')
        .attr('class', 'smoke')
        .attr('cx', function (d) { return x(d.x - 28) })
        .attr('cy', function (d) { return y(d.y - 2) })
        .attr('r', '10px')
        .attr('opacity', '0.8')
        .style('fill', function (d) { return colorScaleEmissions(factoryEmissions[ind][d['name']]) })

    if (levels > 0) {
        scatterPlotSvg.append("g")
            .selectAll("path")
            .data(smokeLocations)
            .enter()
            .append('circle')
            .attr('class', 'smoke')
            .attr('cx', function (d) { return x(d.x - 28.5) })
            .attr('cy', function (d) { return y(d.y - 1) })
            .attr('r', '10px')
            .attr('opacity', '0.8')
            .style('fill', function (d) { return colorScaleEmissions(factoryEmissions[ind][d['name']]) })

        scatterPlotSvg.append("g")
            .selectAll("path")
            .data(smokeLocations)
            .enter()
            .append('circle')
            .attr('class', 'smoke')
            .attr('cx', function (d) { return x(d.x - 27.5) })
            .attr('cy', function (d) { return y(d.y - 1) })
            .attr('r', '10px')
            .attr('opacity', '0.8')
            .style('fill', function (d) { return colorScaleEmissions(factoryEmissions[ind][d['name']]) })
    }

    if (levels > 1) {
        scatterPlotSvg.append("g")
            .selectAll("path")
            .data(smokeLocations)
            .enter()
            .append('circle')
            .attr('class', 'smoke')
            .attr('cx', function (d) { return x(d.x - 28.5) })
            .attr('cy', function (d) { return y(d.y) })
            .attr('r', '10px')
            .attr('opacity', '0.8')
            .style('fill', function (d) { return colorScaleEmissions(factoryEmissions[ind][d['name']]) })

        scatterPlotSvg.append("g")
            .selectAll("path")
            .data(smokeLocations)
            .enter()
            .append('circle')
            .attr('class', 'smoke')
            .attr('cx', function (d) { return x(d.x - 27.5) })
            .attr('cy', function (d) { return y(d.y) })
            .attr('r', '10px')
            .attr('opacity', '0.8')
            .style('fill', function (d) { return colorScaleEmissions(factoryEmissions[ind][d['name']]) })

        scatterPlotSvg.append("g")
            .selectAll("path")
            .data(smokeLocations)
            .enter()
            .append('circle')
            .attr('class', 'smoke')
            .attr('cx', function (d) { return x(d.x - 29) })
            .attr('cy', function (d) { return y(d.y) })
            .attr('r', '10px')
            .attr('opacity', '0.8')
            .style('fill', function (d) { return colorScaleEmissions(factoryEmissions[ind][d['name']]) })

        scatterPlotSvg.append("g")
            .selectAll("path")
            .data(smokeLocations)
            .enter()
            .append('circle')
            .attr('class', 'smoke')
            .attr('cx', function (d) { return x(d.x - 27) })
            .attr('cy', function (d) { return y(d.y) })
            .attr('r', '10px')
            .attr('opacity', '0.8')
            .style('fill', function (d) { return colorScaleEmissions(factoryEmissions[ind][d['name']]) })
    }

}