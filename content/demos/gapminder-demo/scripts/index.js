
var margin = { top: 10, right: 40, bottom: 40, left: 60 }

var width = 1080 - margin.left - margin.right;
var height = 720 - margin.top - margin.bottom;

var svg = d3.select("#my_dataviz")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");;

var tooltip = d3.select("body")
    .append("div")
    .attr('class', 'tooltip')
    .style("opacity", 0);

// filters
var year;
var xAttribute;
var yAttribute;
var region;

var xAttributeDisplayValue;
var yAttributeDisplayValue;

// data
var xData;
var yData;
var xDataFiltered;
var yDataFiltered;
var data = [];

// svg elements
var gDots;
var yearLabel;

// axes
var x = d3.scaleLinear().range([0, width]);
var y = d3.scaleLinear().range([height, 0]);

const t = d3.transition().duration(100)

const colorScale = d3.scaleOrdinal()
    .range(d3.schemeCategory10);

var running = false;
var timer;

document.addEventListener('DOMContentLoaded', function () {
    tooltip.style("opacity", 0);
    d3.select("#year-input").on("input", function () {
        year = document.getElementById("year-input").value;
        document.getElementById('slider').value = year
        yearLabel.text(year);
        loadXYDataFiltered()
    });
    d3.select("#x-attribute").on("input", function () {
        loadXData()
    });
    d3.select("#y-attribute").on("input", function () {
        loadYData()
    });
    d3.select("#regions").on("input", function () {
        region = document.getElementById("regions").value;
        loadXYDataFiltered()
    });

    d3.select("#play-button").on("click", function () {
        onPlayButtonClicked()
    });

    d3.select("#slider").on("input", function () {
        onSliderChanged()
    });

    loadRegions()
});

function drawInitialViz() {
    getInitialFilterValues()
    drawScatterPlot()
    loadXData()
    loadYData()
}

function getInitialFilterValues() {
    year = document.getElementById("year-input").value;
    region = document.getElementById("regions").value;
    xAttribute = document.getElementById("x-attribute").value;
    yAttribute = document.getElementById("y-attribute").value;
}

function loadXData() {
    var selected = document.getElementById("x-attribute");
    xAttributeDisplayValue = selected.options[selected.selectedIndex].text;

    xAttribute = selected.value;
    d3.csv('data/processed/' + xAttribute + '.csv').then(function (values) {
        xData = values;

        x.domain([0, d3.max(xData, function (d) {
            return +d["value"];
        })]);

        loadXDataFiltered()

        drawScatterPlot()
    });
}

function loadYData() {
    var selected = document.getElementById("y-attribute");
    yAttributeDisplayValue = selected.options[selected.selectedIndex].text;

    yAttribute = selected.value;

    d3.csv('data/processed/' + yAttribute + '.csv').then(function (values) {
        yData = values;

        y.domain([0, d3.max(yData, function (d) {
            return +d["value"];;
        })]);

        loadYDataFiltered()

        drawScatterPlot()
    });
}

function loadXYDataFiltered() {
    loadXDataFiltered()
    loadYDataFiltered()
}

function loadXDataFiltered() {
    xDataFiltered = xData.filter((value) => {
        return value["region"] == region && value["year"] == year
    });

    constructScatterData()
}

function loadYDataFiltered() {
    yDataFiltered = yData.filter((value) => {
        return value["region"] == region && value["year"] == year
    });

    constructScatterData()
}

function constructScatterData() {
    if (xDataFiltered == undefined || yDataFiltered == undefined) {
        return
    }
    data = xDataFiltered.map((val) => {
        var yFiltered = yDataFiltered.filter((v) => v.country == val.country)
        if (yFiltered.length == 0) {
            return null;
        }

        return {
            "country": val.country,
            "geo": val.geo,
            "x": +val.value,
            "y": +yFiltered[0]['value'],
            "region": val.region
        }
    }).filter((v) => v !== null)

    drawCircles()
}

// Draw the map in the #map svg
function drawScatterPlot() {
    svg.selectAll("*").remove()

    svg.append("text")
        .attr("x", (width / 2))
        .attr("y", 10 - (margin.top / 2))
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .attr("font-family", "sans-serif")
        .style("text-decoration", "underline")
        .attr("font-weight", 700)
        .text(region + ": " + yAttributeDisplayValue + " vs " + xAttributeDisplayValue);

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    svg.append("g")
        .call(d3.axisLeft(y));

    svg.append("text")
        .attr("transform",
            "translate(" + (width / 2) + " ," +
            (height + margin.top + 20) + ")")
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .attr("font-family", "sans-serif")
        .attr("font-weight", 400)
        .text(xAttributeDisplayValue);

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -60)
        .attr("x", 0 - (height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .attr("font-family", "sans-serif")
        .attr("font-weight", 400)
        .text(yAttributeDisplayValue);

    yearLabel = svg.append("text")
        .attr("class", "year label")
        .attr("text-anchor", "end")
        .attr("y", height - 24)
        .attr("x", width)
        .text(year);

    gDots = svg.append("g")
    // Add dots
    drawCircles()
}

function drawCircles() {
    gDots.selectAll('g')
        .data(data)
        .join(
            enter => enterCircles(enter, t),
            update => updateCircles(update, t),
            exit => exitCircles(exit, t)
        )

    gDots.selectAll('circle')
        .on('mouseover', function (d, i) {
            tooltip.style("opacity", 1);
            tooltip.html(d.country)
                .style("left", (d3.event.pageX + 10) + "px")
                .style("top", (d3.event.pageY - 15) + "px");
        })
        .on('mouseout', function (d, i) {
            tooltip.style("opacity", 0);
        })
}

function enterCircles(enter, t) {
    enter.append('g')
        .call(g =>
            g
                .append("circle")
                .attr("class", "circles")
                .attr("r", 18)
                .transition(t)
                .attr("cx", function (d) { return x(d.x); })
                .attr("cy", function (d) { return y(d.y); })
                .style("fill", function (d) { return colorScale(d.region) })
                .attr('stroke', '#000000')
                .attr('stroke-width', 1)
                .attr("opacity", 0.8)
        )
        .call(g =>
            g.append('text')
                .style('fill', 'black')
                .attr("text-anchor", "middle")
                .transition(t)
                .text(function (d) {
                    return d.geo;
                })
                .attr("x", function (d) {
                    return x(d.x);
                })
                .attr("y", function (d) {
                    return y(d.y);
                })
        )
}

function updateCircles(update, t) {
    update
        .call(g => g.select('text')
            .attr("text-anchor", "middle")
            .transition(t)
            .text(function (d) {
                return d.geo;
            })
            .attr("x", function (d) {
                return x(d.x);
            })
            .attr("y", function (d) {
                return y(d.y);
            })
        )
        .call(g => g.select('circle')
            .transition(t)
            .attr("cx", function (d) { return x(d.x); })
            .attr("cy", function (d) { return y(d.y); })
        )
}

function exitCircles(exit, t) {
    exit
        .call(exit => exit
            .transition()
            .duration(200)
            .ease(d3.easeCubic)
            .attr("r", 0)
            .remove())
}

function loadRegions() {
    d3.csv("data/countries_regions.csv").then(function (data) {
        data = data.map(function (item) {
            return item['World bank region']
        })

        var select = document.getElementById("regions");
        return [...new Set(data)].forEach(element => {
            select.options[select.options.length] = new Option(element, element);
        });
    }).then(function (data) {
        drawInitialViz()
    })
}

function onPlayButtonClicked() {
    var duration = 500,
        maxstep = 2020;

    if (running == true) {
        document.getElementById('play-button-span').innerHTML = "Play"
        running = false;
        clearInterval(timer);
    } else if (running == false) {
        document.getElementById('play-button-span').innerHTML = "Pause"
        sliderValue = document.getElementById('slider').value;

        timer = setInterval(function () {
            if (sliderValue < maxstep) {
                sliderValue++;
                document.getElementById('slider').value = sliderValue
                document.getElementById('year-input').value = sliderValue
                year = sliderValue
                yearLabel.text(year);
            }
            document.getElementById('slider').value = sliderValue
            loadXYDataFiltered();

        }, duration);
        running = true;
    }
}

function onSliderChanged() {
    document.getElementById('year-input').value = document.getElementById('slider').value
    year = document.getElementById('slider').value
    yearLabel.text(year);
    clearInterval(timer);
    document.getElementById('play-button-span').innerHTML = "Play"
    loadXYDataFiltered()
}