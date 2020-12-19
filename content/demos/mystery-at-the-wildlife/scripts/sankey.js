// set the dimensions and margins of the graph
const margin = {top: 20, right: 40, bottom: 20, left: 40};
const outerWidth = 600;
const outerHeight = 520;
const width = outerWidth - margin.left - margin.right;
const height = outerHeight - margin.top - margin.bottom;

const colorScale =  d3.scaleSequential()
                  .interpolator(d3.interpolateOrRd);

// div for tooltip
var div = d3.select("body")
		    .append("div")
		    .attr("class", "tooltip")
			.style("opacity", 0);                  

d3.json("data/sankey.json").then(function (data) {

    var svg = d3.select("#sankey").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

    var color = d3.scaleOrdinal(d3.schemeCategory10);

    var sankey = d3.sankey()
        .nodeWidth(20)
        .nodePadding(10)
        .size([width, height]);

    sankey
        .nodes(data.nodes)
        .links(data.links)
        .layout(1);

    var arr = []
    data.links.forEach(function (value, index, array) {
        arr.push(parseInt(value.value));
      })
    max_val = Math.max(...arr)
    min_val = Math.min(...arr)

    colorScale.domain([min_val, max_val])

    var link = svg.append("g")
        .selectAll(".link")
        .data(data.links)
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", sankey.link())
        .attr('opacity', 0.2)
        .attr('stroke', function(d) { return d.color = colorScale(d.value); })
        .attr("stroke-width", function(d) { return Math.max(1, d.dy); })
        .on("mouseover",function(d){
            if (d3.event.defaultPrevented) return;
            tooltipHtml = "Value: " + d.value
            div.transition()
				.duration(50)
				.style("opacity", 1);

			div.html(tooltipHtml)
				.style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY) + "px");
                
        })
        .on("mouseout",function(d){
            if (d3.event.defaultPrevented) return;
            div.transition()
				.duration(50)
				.style("opacity", 0);
                
        })
        .sort(function(a, b) { return b.dy - a.dy; });

    var node = svg.append("g")
        .selectAll(".node")
        .data(data.nodes)
        .enter().append("g")
        .attr("class", "node")
        .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
        .call(d3.drag()
            .subject(function(d) { return d; })
            .on("start", function() { this.parentNode.appendChild(this); }))

    node
        .append("rect")
        .attr("height", function(d) { return d.dy; })
        .attr("width", sankey.nodeWidth())
        .style("fill", function(d) { return d.color = color(d.name.replace(/ .*/, "")); })
        .style("stroke", function(d) { return d3.rgb(d.color).darker(2); })
        .append("title")
        .text(function(d) { return d.name + "\n" + "There is " + d.value + " stuff in this node"; });

    node
        .append("text")
        .attr("x", -6)
        .attr("y", function(d) { return d.dy / 2; })
        .attr("dy", ".35em")
        .attr("text-anchor", "end")
        .attr("transform", null)
        .text(function(d) { return d.name; })
        .filter(function(d) { return d.x < width / 2; })
        .attr("x", 6 + sankey.nodeWidth())
        .attr("text-anchor", "start");
});