import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { StepMetric } from '../types'

interface LossChartProps {
  data: StepMetric[]
  currentStep?: number
}

function LossChart({ data, currentStep }: LossChartProps) {
  if (data.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        No step data yet.
      </div>
    )
  }

  // Format data for chart
  const chartData = data.map(point => ({
    step: point.step,
    loss: point.loss,
    pgLoss: point.pg_loss || 0,
    klLoss: point.kl_loss || 0,
    gradNorm: point.grad_norm || 0,
  }))

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis 
          dataKey="step" 
          stroke="#9ca3af" 
          tick={{ fill: '#9ca3af', fontSize: 10 }}
        />
        <YAxis 
          stroke="#9ca3af" 
          tick={{ fill: '#9ca3af', fontSize: 10 }}
          tickFormatter={(v) => v.toFixed(2)}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: '#1f2937', 
            border: '1px solid #374151',
            borderRadius: '0.5rem',
          }}
          labelStyle={{ color: '#9ca3af' }}
          formatter={(value: number) => value.toFixed(4)}
        />
        {/* Current step indicator */}
        {currentStep && currentStep > 0 && (
          <ReferenceLine 
            x={currentStep} 
            stroke="#f59e0b" 
            strokeWidth={2}
            strokeDasharray="4 4"
            label={{ value: 'Now', fill: '#f59e0b', fontSize: 10 }}
          />
        )}
        <Line
          type="monotone"
          dataKey="loss"
          stroke="#ef4444"
          strokeWidth={2}
          dot={false}
          name="Total Loss"
        />
        <Line
          type="monotone"
          dataKey="pgLoss"
          stroke="#3b82f6"
          strokeWidth={1}
          dot={false}
          name="PG Loss"
          opacity={0.7}
        />
        <Line
          type="monotone"
          dataKey="klLoss"
          stroke="#8b5cf6"
          strokeWidth={1}
          dot={false}
          name="KL Loss"
          opacity={0.7}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default LossChart
