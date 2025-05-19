import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Box,
  Card,
  CardContent,
  Tooltip,
  AppBar,
  Toolbar,
  IconButton,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import axios from 'axios';
import GitHubIcon from '@mui/icons-material/GitHub';
import { styled } from '@mui/material/styles';

const API_BASE_URL = 'http://localhost:5001/api';

// Color palette for charts
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

// Add formula definitions
const STAT_FORMULAS = {
  mean: "Mean = Σ(x) / n\nWhere:\n- Σ(x) is the sum of all values\n- n is the total number of values",
  median: "Median = Middle value of ordered data\nFor even number of values:\nMedian = (n/2th value + (n/2+1)th value) / 2",
  std_dev: "Standard Deviation = √(Σ(x - μ)² / n)\nWhere:\n- μ is the mean\n- x is each value\n- n is the total number of values",
  min: "Minimum = Lowest value in the dataset",
  max: "Maximum = Highest value in the dataset",
  q1: "Q1 (First Quartile) = Median of lower half of data",
  q3: "Q3 (Third Quartile) = Median of upper half of data",
  iqr: "IQR (Interquartile Range) = Q3 - Q1\nWhere:\n- Q3 is the third quartile\n- Q1 is the first quartile"
};

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  background: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  borderRadius: '15px',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  transition: 'transform 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-5px)',
  },
}));

function App() {
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState('');
  const [stations, setStations] = useState([]);
  const [selectedStation, setSelectedStation] = useState('');
  const [data, setData] = useState([]);
  const [stats, setStats] = useState(null);
  const [stationComparison, setStationComparison] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch states
  useEffect(() => {
    const fetchStates = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/states`);
        setStates(response.data);
        if (response.data.length > 0) {
          setSelectedState(response.data[0]);
        }
        setError(null);
      } catch (err) {
        console.error('Error fetching states:', err);
        setError('Failed to fetch states. Please make sure the backend server is running.');
      } finally {
        setLoading(false);
      }
    };

    fetchStates();
  }, []);

  // Fetch stations when state changes
  useEffect(() => {
    const fetchStations = async () => {
      if (!selectedState) return;

      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/stations?state=${selectedState}`);
        setStations(response.data);
        if (response.data.length > 0) {
          setSelectedStation(response.data[0]);
        }
        setError(null);
      } catch (err) {
        console.error('Error fetching stations:', err);
        setError('Failed to fetch stations');
      } finally {
        setLoading(false);
      }
    };

    fetchStations();
  }, [selectedState]);

  // Fetch data when station changes
  useEffect(() => {
    const fetchData = async () => {
      if (!selectedStation) return;

      try {
        setLoading(true);
        const [dataResponse, comparisonResponse] = await Promise.all([
          axios.get(`${API_BASE_URL}/data?state=${selectedState}&station=${selectedStation}`),
          axios.get(`${API_BASE_URL}/station_comparison?state=${selectedState}`)
        ]);

        if (dataResponse.data.error) {
          throw new Error(dataResponse.data.error);
        }
        if (comparisonResponse.data.error) {
          throw new Error(comparisonResponse.data.error);
        }

        setData(dataResponse.data.data || []);
        setStats(dataResponse.data.stats || null);
        setStationComparison(comparisonResponse.data || []);
        setError(null);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.response?.data?.error || err.message || 'Failed to fetch data. Please try again.');
        setData([]);
        setStats(null);
        setStationComparison([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedState, selectedStation]);

  const handleStateChange = (event) => {
    setSelectedState(event.target.value);
  };

  const handleStationChange = (event) => {
    setSelectedStation(event.target.value);
  };

  // Calculate temperature distribution data
  const calculateDistribution = (data) => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return [];
    }

    const temps = [...data.map(d => d.predicted), ...data.map(d => d.actual)];
    const min = Math.floor(Math.min(...temps));
    const max = Math.ceil(Math.max(...temps));
    const range = max - min;
    const binSize = Math.ceil(range / 10);

    const distribution = [];
    for (let i = min; i <= max; i += binSize) {
      const count = temps.filter(t => t >= i && t < i + binSize).length;
      distribution.push({
        range: `${i}°F - ${i + binSize}°F`,
        count: count
      });
    }
    return distribution;
  };

  // Calculate daily temperature range
  const calculateDailyRange = (data) => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return [];
    }

    return data.map(d => ({
      date: d.date,
      range: Math.abs(d.predicted - d.actual),
      predicted: d.predicted,
      actual: d.actual
    }));
  };

  if (loading && !data?.length) {
    return (
      <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <CircularProgress size={60} sx={{ color: '#21CBF3' }} />
        <Typography variant="h6" sx={{ mt: 2, color: 'white' }}>
          Loading Dashboard...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <Alert severity="error" sx={{ width: '80%', maxWidth: '500px' }}>{error}</Alert>
      </Box>
    );
  }

  return (
    <>
      <AppBar
        position="fixed"
        sx={{
          background: 'rgba(0, 4, 40, 0.8)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
        }}
      >
        <Toolbar>
          <Typography
            variant="h6"
            component="div"
            sx={{
              flexGrow: 1,
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              fontWeight: 'bold'
            }}
          >
            BlizzardX
          </Typography>
          <IconButton
            color="inherit"
            aria-label="github"
            onClick={() => window.open('https://github.com/loki-k4/BlizzardX', '_blank', 'noopener,noreferrer')}
          >
            <GitHubIcon />
          </IconButton>
        </Toolbar>
      </AppBar>
      <Toolbar /> {/* Spacer for fixed AppBar */}
      <Container
        maxWidth={false}
        disableGutters
        sx={{
          py: 4,
          minHeight: '100vh',
          position: 'relative',
          mt: 2,
          px: '10%'
        }}
      >
        <Typography
          variant="h2"
          component="h1"
          gutterBottom
          align="center"
          sx={{
            fontWeight: 'bold',
            mb: 6, // Increase bottom margin
            textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
            background: 'linear-gradient(90deg, rgba(0, 0, 51, 1) 0%, rgba(0, 0, 255, 1) 35%, rgba(0, 255, 255, 1) 70%, rgba(0, 0, 0, 1) 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '20px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            boxShadow: '0 8px 32px 0 rgba(0, 4, 40, 0.3)',
            position: 'relative',
            zIndex: 1,
            animation: 'fadeIn 1s ease-in',
            '@keyframes fadeIn': {
              '0%': {
                opacity: 0,
                transform: 'translateY(-20px)'
              },
              '100%': {
                opacity: 1,
                transform: 'translateY(0)'
              }
            }
          }}
        >
          BlizzardX Dashboard
        </Typography>

        {/* State and Station Selection with Statistics */}
        <Grid container spacing={3} sx={{ mb: 6, alignItems: 'center' }} id="temp-stats">
          {/* State Dropdown with Label */}
          <Grid item xs={12} md={6}>
            <Grid container alignItems="center" spacing={1}>
              <Grid item>
                <Typography sx={{ color: 'white', fontWeight: 500, minWidth: 50 }}>State</Typography>
              </Grid>
              <Grid item xs>
                <FormControl fullWidth>
                  <Select
                    value={selectedState}
                    onChange={handleStateChange}
                    sx={{
                      color: 'white',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.3)'
                      },
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.5)'
                      },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#21CBF3'
                      },
                      height: '56px'
                    }}
                  >
                    {states.map(state => (
                      <MenuItem key={state} value={state}>{state}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Grid>
          {/* Station Dropdown with Label */}
          <Grid item xs={12} md={6}>
            <Grid container alignItems="center" spacing={1}>
              <Grid item>
                <Typography sx={{ color: 'white', fontWeight: 500, minWidth: 60 }}>Station</Typography>
              </Grid>
              <Grid item xs>
                <FormControl fullWidth>
                  <Select
                    value={selectedStation}
                    onChange={handleStationChange}
                    sx={{
                      color: 'white',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.3)'
                      },
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(255, 255, 255, 0.5)'
                      },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#21CBF3'
                      },
                      height: '56px'
                    }}
                  >
                    {stations.map(station => (
                      <MenuItem key={station} value={station}>{station}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Grid>
          {/* Temperature Statistics below, full width */}
          <Grid item xs={12}>
            <Paper sx={{
              p: 3,
              background: 'rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(10px)',
              borderRadius: '16px',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              boxShadow: '0 8px 32px 0 rgba(0, 4, 40, 0.3)',
              mt: 2
            }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'white', textAlign: 'center' }}>
                Temperature Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Typography variant="subtitle2" sx={{ color: 'white' }}>Predicted Mean</Typography>
                  <Typography variant="h6" sx={{ color: 'white' }}>{stats?.predicted_mean.toFixed(1)}°F</Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="subtitle2" sx={{ color: 'white' }}>Actual Mean</Typography>
                  <Typography variant="h6" sx={{ color: 'white' }}>{stats?.actual_mean.toFixed(1)}°F</Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="subtitle2" sx={{ color: 'white' }}>RMSE</Typography>
                  <Typography variant="h6" sx={{ color: 'white' }}>{stats?.rmse.toFixed(1)}°F</Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="subtitle2" sx={{ color: 'white' }}>MAE</Typography>
                  <Typography variant="h6" sx={{ color: 'white' }}>{stats?.mae.toFixed(1)}°F</Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>

        {loading ? (
          <Box display="flex" justifyContent="center" my={4}>
            <CircularProgress />
          </Box>
        ) : (
          <Grid container spacing={4}>
            {/* Time Series Chart */}
            <Grid item xs={12} id="time-series">
              <Paper sx={{
                p: 4,
                background: 'rgba(255, 255, 255, 0.1)',
                backdropFilter: 'blur(10px)',
                borderRadius: '16px',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                boxShadow: '0 8px 32px 0 rgba(0, 4, 40, 0.3)',
                transition: 'all 0.3s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-5px)',
                  boxShadow: '0 12px 40px 0 rgba(0, 4, 40, 0.5)'
                },
                mb: 4,
                width: '100%',
                mx: 'auto'
              }}>
                <Typography
                  variant="h5"
                  gutterBottom
                  sx={{
                    color: 'white',
                    textAlign: 'center',
                    mb: 3,
                    fontWeight: 'bold',
                    textShadow: '2px 2px 4px rgba(0,0,0,0.3)'
                  }}
                >
                  Predicted vs Actual Temperature Over Time
                </Typography>
                {data && data.length > 0 ? (
                  <ResponsiveContainer width="100%" height={600}>
                    <LineChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={(date) => new Date(date).toLocaleDateString()}
                        stroke="white"
                        tick={{ fill: 'white', fontSize: 12 }}
                        label={{
                          value: 'Date',
                          position: 'bottom',
                          fill: 'white',
                          style: { fontSize: 14, fontWeight: 'bold' }
                        }}
                      />
                      <YAxis
                        label={{
                          value: 'Temperature (°F)',
                          angle: -90,
                          position: 'insideLeft',
                          fill: 'white',
                          style: { fontSize: 14, fontWeight: 'bold' }
                        }}
                        stroke="white"
                        tick={{ fill: 'white', fontSize: 12 }}
                      />
                      <Tooltip
                        labelFormatter={(date) => new Date(date).toLocaleDateString()}
                        formatter={(value) => [`${value.toFixed(1)}°F`, '']}
                        contentStyle={{
                          background: 'rgba(255, 255, 255, 0.1)',
                          backdropFilter: 'blur(10px)',
                          border: '1px solid rgba(255, 255, 255, 0.2)',
                          borderRadius: '8px',
                          fontSize: '14px'
                        }}
                        labelStyle={{ color: 'white', fontSize: '14px' }}
                      />
                      <Legend
                        wrapperStyle={{
                          color: 'white',
                          fontSize: '14px',
                          padding: '20px'
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#8884d8"
                        name="Predicted Temperature"
                        strokeWidth={3}
                        dot={{ r: 4, strokeWidth: 2 }}
                        activeDot={{ r: 8, strokeWidth: 2 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#82ca9d"
                        name="Actual Temperature"
                        strokeWidth={3}
                        dot={{ r: 4, strokeWidth: 2 }}
                        activeDot={{ r: 8, strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <Box
                    display="flex"
                    justifyContent="center"
                    alignItems="center"
                    height={600}
                  >
                    <Typography variant="body1" sx={{ color: 'white' }}>
                      No data available for the selected station
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>

            <Box display="flex" gap={4} mb={6} sx={{ width: '100%', flexWrap: 'wrap', background: 'transparent' }}>
              <Box flex={1} minWidth={320} maxWidth="100%" boxSizing="border-box" overflow="hidden" id="temp-dist">
                <Paper sx={{
                  p: 4,
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px 0 rgba(0, 4, 40, 0.3)',
                  height: '100%',
                  overflow: 'visible',
                  mb: 4
                }}>
                  <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                    Temperature Distribution
                  </Typography>
                  {data && data.length > 0 ? (
                    <div style={{ borderRadius: '16px', overflow: 'hidden' }}>
                      <ResponsiveContainer width="100%" height={400}>
                        <BarChart data={calculateDistribution(data)}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                          <XAxis
                            dataKey="range"
                            stroke="white"
                            angle={-45}
                            textAnchor="end"
                            height={70}
                            interval={0}
                            tick={{ fill: 'white', fontSize: 10 }}
                          />
                          <YAxis
                            label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: 'white' }}
                            stroke="white"
                            tick={{ fill: 'white' }}
                          />
                          <Tooltip
                            contentStyle={{
                              background: 'rgba(255, 255, 255, 0.1)',
                              backdropFilter: 'blur(10px)',
                              border: '1px solid rgba(255, 255, 255, 0.2)',
                              borderRadius: '8px'
                            }}
                            labelStyle={{ color: 'white' }}
                          />
                          <Bar dataKey="count" fill="#8884d8" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <Box
                      display="flex"
                      justifyContent="center"
                      alignItems="center"
                      height={400}
                    >
                      <Typography variant="body1" sx={{ color: 'white' }}>
                        No data available for the selected station
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Box>
              <Box flex={1} minWidth={320} maxWidth="100%" boxSizing="border-box" overflow="hidden" id="pred-accuracy">
                <Paper sx={{
                  p: 4,
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px 0 rgba(0, 4, 40, 0.3)',
                  height: '100%',
                  overflow: 'visible',
                  mb: 4
                }}>
                  <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                    Prediction Accuracy
                  </Typography>
                  {data && data.length > 0 ? (
                    <div style={{ borderRadius: '16px', overflow: 'hidden' }}>
                      <ResponsiveContainer width="100%" height={400}>
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                          <XAxis
                            dataKey="actual"
                            name="Actual"
                            stroke="white"
                            label={{ value: 'Actual Temperature (°F)', position: 'bottom', fill: 'white' }}
                            tick={{ fill: 'white' }}
                          />
                          <YAxis
                            dataKey="predicted"
                            name="Predicted"
                            stroke="white"
                            label={{ value: 'Predicted Temperature (°F)', angle: -90, position: 'insideLeft', fill: 'white' }}
                            tick={{ fill: 'white' }}
                          />
                          <Tooltip
                            contentStyle={{
                              background: 'rgba(255, 255, 255, 0.1)',
                              backdropFilter: 'blur(10px)',
                              border: '1px solid rgba(255, 255, 255, 0.2)',
                              borderRadius: '8px'
                            }}
                            labelStyle={{ color: 'white' }}
                          />
                          <Scatter
                            data={data}
                            fill="#8884d8"
                            fillOpacity={0.6}
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <Box
                      display="flex"
                      justifyContent="center"
                      alignItems="center"
                      height={400}
                    >
                      <Typography variant="body1" sx={{ color: 'white' }}>
                        No data available for the selected station
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Box>
            </Box>

            {/* Bottom row: Daily Temperature Range & Station Comparison */}
            <Box display="flex" gap={4} mt={2} sx={{ width: '100%', flexWrap: 'wrap', background: 'transparent' }}>
              <Box flex={1} minWidth={320} maxWidth="100%" boxSizing="border-box" overflow="hidden" id="daily-range">
                <Paper sx={{
                  p: 4,
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px 0 rgba(0, 4, 40, 0.3)',
                  height: '100%',
                  overflow: 'visible',
                  mt: 2
                }}>
                  <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                    Daily Temperature Range
                  </Typography>
                  {data && data.length > 0 ? (
                    <div style={{ borderRadius: '16px', overflow: 'hidden' }}>
                      <ResponsiveContainer width="100%" height={400}>
                        <AreaChart data={calculateDailyRange(data)}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                          <XAxis
                            dataKey="date"
                            stroke="white"
                            tickFormatter={(date) => new Date(date).toLocaleDateString()}
                            tick={{ fill: 'white' }}
                          />
                          <YAxis
                            stroke="white"
                            label={{ value: 'Temperature (°F)', angle: -90, position: 'insideLeft', fill: 'white' }}
                            tick={{ fill: 'white' }}
                          />
                          <Tooltip
                            labelFormatter={(date) => new Date(date).toLocaleDateString()}
                            formatter={(value) => [`${value.toFixed(1)}°F`, '']}
                            contentStyle={{
                              background: 'rgba(255, 255, 255, 0.1)',
                              backdropFilter: 'blur(10px)',
                              border: '1px solid rgba(255, 255, 255, 0.2)',
                              borderRadius: '8px'
                            }}
                            labelStyle={{ color: 'white' }}
                          />
                          <Legend wrapperStyle={{ color: 'white' }} />
                          <Area
                            type="monotone"
                            dataKey="range"
                            stroke="#8884d8"
                            fill="#8884d8"
                            fillOpacity={0.3}
                            name="Temperature Range"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <Box
                      display="flex"
                      justifyContent="center"
                      alignItems="center"
                      height={400}
                    >
                      <Typography variant="body1" sx={{ color: 'white' }}>
                        No data available for the selected station
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Box>
              <Box flex={1} minWidth={320} maxWidth="100%" boxSizing="border-box" overflow="hidden" id="station-comparison">
                <Paper sx={{
                  p: 4,
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '16px',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px 0 rgba(0, 4, 40, 0.3)',
                  height: '100%',
                  overflow: 'visible',
                  mt: 2
                }}>
                  <Typography variant="h6" gutterBottom sx={{ color: 'white' }}>
                    Station Comparison (RMSE)
                  </Typography>
                  {stationComparison && stationComparison.length > 0 ? (
                    <div style={{ borderRadius: '16px', overflow: 'hidden' }}>
                      <ResponsiveContainer width="100%" height={400}>
                        <BarChart data={stationComparison}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                          <XAxis
                            dataKey="Station_ID"
                            stroke="white"
                            angle={-45}
                            textAnchor="end"
                            height={70}
                            interval={0}
                            tick={{ fill: 'white', fontSize: 10 }}
                          />
                          <YAxis
                            label={{ value: 'RMSE (°F)', angle: -90, position: 'insideLeft', fill: 'white' }}
                            stroke="white"
                            tick={{ fill: 'white' }}
                          />
                          <Tooltip
                            formatter={(value) => [`${value.toFixed(1)}°F`, 'RMSE']}
                            contentStyle={{
                              background: 'rgba(255, 255, 255, 0.1)',
                              backdropFilter: 'blur(10px)',
                              border: '1px solid rgba(255, 255, 255, 0.2)',
                              borderRadius: '8px'
                            }}
                            labelStyle={{ color: 'white' }}
                          />
                          <Legend wrapperStyle={{ color: 'white' }} />
                          <Bar
                            dataKey="rmse"
                            fill="#8884d8"
                            name="RMSE"
                            radius={[4, 4, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <Box
                      display="flex"
                      justifyContent="center"
                      alignItems="center"
                      height={400}
                    >
                      <Typography variant="body1" sx={{ color: 'white' }}>
                        No station comparison data available
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Box>
            </Box>
          </Grid>
        )}
      </Container>
    </>
  );
}

export default App;
